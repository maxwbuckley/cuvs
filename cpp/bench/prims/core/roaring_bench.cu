/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: GPU Roaring Bitmap set operations and decompression
 * for prefiltered vector search in cuVS.
 *
 * Compares:
 *   - GPU Roaring AND vs flat bitset AND (thrust)
 *   - GPU Roaring decompress time
 *   - Multi-predicate filter construction (AND chain)
 *
 * Build standalone:
 *   cd cpp/bench/prims/core
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
 *   make -j
 *   ./roaring_bench
 */

#include <cuvs/core/roaring.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// Generate sorted, deduplicated IDs with given density
static std::vector<uint32_t> generate_ids(uint32_t universe, double density, uint64_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<uint32_t> ids;
  ids.reserve(static_cast<size_t>(universe * density * 1.1));
  for (uint32_t i = 0; i < universe; ++i) {
    if (dist(gen) < density) ids.push_back(i);
  }
  return ids;
}

// Flat bitset AND on GPU using a simple kernel
__global__ void flat_bitset_and_kernel(const uint32_t* a,
                                       const uint32_t* b,
                                       uint32_t* out,
                                       uint32_t n_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_words) { out[idx] = a[idx] & b[idx]; }
}

struct BenchResult {
  const char* name;
  double median_ms;
  double mean_ms;
  double stddev_ms;
  double min_ms;
  double max_ms;
};

static BenchResult run_timed(const char* name,
                             int warmup,
                             int iterations,
                             std::function<void()> fn)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i)
    fn();
  cudaDeviceSynchronize();

  std::vector<double> times(iterations);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < iterations; ++i) {
    cudaEventRecord(start);
    fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    times[i] = ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::sort(times.begin(), times.end());
  double median = times[iterations / 2];
  double sum    = 0;
  for (auto t : times)
    sum += t;
  double mean = sum / iterations;
  double var  = 0;
  for (auto t : times)
    var += (t - mean) * (t - mean);
  double stddev = std::sqrt(var / iterations);

  return {name, median, mean, stddev, times.front(), times.back()};
}

static void print_result(const BenchResult& r)
{
  printf("  %-45s  median=%.3f ms  mean=%.3f ms  std=%.3f ms  [%.3f, %.3f]\n",
         r.name,
         r.median_ms,
         r.mean_ms,
         r.stddev_ms,
         r.min_ms,
         r.max_ms);
}

int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Print GPU info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB)\n\n", prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0));

  // Test configurations
  struct Config {
    const char* name;
    uint32_t universe;
    double density_a;
    double density_b;
  };

  Config configs[] = {
    {"1M Dense×Dense (50%×30%)", 1000000, 0.5, 0.3},
    {"10M Dense×Dense (50%×30%)", 10000000, 0.5, 0.3},
    {"100M Dense×Dense (50%×30%)", 100000000, 0.5, 0.3},
    {"1B Dense×Dense (50%×30%)", 1000000000, 0.5, 0.3},
    {"100M Sparse×Sparse (1%×0.5%)", 100000000, 0.01, 0.005},
    {"1B Sparse×Sparse (1%×0.5%)", 1000000000, 0.01, 0.005},
    {"100M Dense×Sparse (50%×1%)", 100000000, 0.5, 0.01},
    {"1B Dense×Sparse (50%×1%)", 1000000000, 0.5, 0.01},
  };

  constexpr int WARMUP = 5;
  constexpr int ITERS  = 30;

  for (auto& cfg : configs) {
    printf("=== %s (universe=%u) ===\n", cfg.name, cfg.universe);

    // Generate data
    printf("  Generating data...\n");
    auto t0   = std::chrono::high_resolution_clock::now();
    auto ids_a = generate_ids(cfg.universe, cfg.density_a, 42);
    auto ids_b = generate_ids(cfg.universe, cfg.density_b, 123);
    auto t1   = std::chrono::high_resolution_clock::now();
    double gen_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Generated: A=%zu ids (%.1f%%), B=%zu ids (%.1f%%) in %.0f ms\n",
           ids_a.size(),
           100.0 * ids_a.size() / cfg.universe,
           ids_b.size(),
           100.0 * ids_b.size() / cfg.universe,
           gen_ms);

    // Upload to GPU Roaring
    auto gpu_a =
      cuvs::core::from_sorted_ids(res, ids_a.data(), ids_a.size(), cfg.universe);
    auto gpu_b =
      cuvs::core::from_sorted_ids(res, ids_b.data(), ids_b.size(), cfg.universe);

    printf("  GPU Roaring A: %zu bytes (%.2f MB), %u containers\n",
           gpu_a.device_memory_bytes(),
           gpu_a.device_memory_bytes() / (1024.0 * 1024.0),
           gpu_a.n_containers);
    printf("  GPU Roaring B: %zu bytes (%.2f MB), %u containers\n",
           gpu_b.device_memory_bytes(),
           gpu_b.device_memory_bytes() / (1024.0 * 1024.0),
           gpu_b.n_containers);
    printf("  Flat bitset: %.2f MB\n", gpu_a.flat_bitset_bytes() / (1024.0 * 1024.0));
    printf("  Compression ratio: %.1fx\n", gpu_a.compression_ratio());

    // Benchmark: GPU Roaring AND
    auto r_and = run_timed("GPU Roaring AND", WARMUP, ITERS, [&]() {
      auto result = cuvs::core::set_op(res, gpu_a, gpu_b, cuvs::core::roaring_set_op::AND);
    });
    print_result(r_and);

    // Benchmark: GPU Roaring decompress
    auto r_decompress = run_timed("GPU Roaring decompress→bitset", WARMUP, ITERS, [&]() {
      auto bs = cuvs::core::to_bitset(res, gpu_a);
    });
    print_result(r_decompress);

    // Benchmark: GPU Roaring AND + decompress (full pipeline)
    auto r_pipeline = run_timed("GPU Roaring AND+decompress", WARMUP, ITERS, [&]() {
      auto combined = cuvs::core::set_op(res, gpu_a, gpu_b, cuvs::core::roaring_set_op::AND);
      auto bs       = cuvs::core::to_bitset(res, combined);
    });
    print_result(r_pipeline);

    // Baseline: flat bitset AND on GPU
    uint32_t n_words = (cfg.universe + 31) / 32;
    auto bs_a      = cuvs::core::to_bitset(res, gpu_a);
    auto bs_b      = cuvs::core::to_bitset(res, gpu_b);
    rmm::device_uvector<uint32_t> flat_out(n_words, stream);

    auto r_flat = run_timed("Flat bitset AND (baseline)", WARMUP, ITERS, [&]() {
      uint32_t blocks = (n_words + 255) / 256;
      flat_bitset_and_kernel<<<blocks, 256, 0, stream>>>(
        bs_a.data(), bs_b.data(), flat_out.data(), n_words);
      cudaStreamSynchronize(stream);
    });
    print_result(r_flat);

    // Optimal pipeline: decompress both → flat AND
    auto r_optimal = run_timed("Optimal: decompress(2)+flat AND", WARMUP, ITERS, [&]() {
      auto bsa = cuvs::core::to_bitset(res, gpu_a);
      auto bsb = cuvs::core::to_bitset(res, gpu_b);
      uint32_t blocks = (n_words + 255) / 256;
      flat_bitset_and_kernel<<<blocks, 256, 0, stream>>>(
        bsa.data(), bsb.data(), flat_out.data(), n_words);
      cudaStreamSynchronize(stream);
    });
    print_result(r_optimal);

    // Memory comparison
    double roaring_mb = (gpu_a.device_memory_bytes() + gpu_b.device_memory_bytes()) / (1024.0 * 1024.0);
    double flat_mb = 2.0 * gpu_a.flat_bitset_bytes() / (1024.0 * 1024.0);
    printf("  Memory: 2 Roaring = %.1f MB vs 2 Flat = %.1f MB (%.1fx savings)\n",
           roaring_mb, flat_mb, flat_mb / roaring_mb);

    // Multi-predicate: AND(A, B, C, D) — 4 filters
    if (cfg.universe <= 100000000) {  // Skip for 1B to keep runtime reasonable
      auto ids_c = generate_ids(cfg.universe, 0.4, 456);
      auto ids_d = generate_ids(cfg.universe, 0.7, 789);
      auto gpu_c =
        cuvs::core::from_sorted_ids(res, ids_c.data(), ids_c.size(), cfg.universe);
      auto gpu_d =
        cuvs::core::from_sorted_ids(res, ids_d.data(), ids_d.size(), cfg.universe);

      cuvs::core::gpu_roaring* arr[] = {&gpu_a, &gpu_b, &gpu_c, &gpu_d};
      auto r_multi =
        run_timed("GPU Roaring multi_and(4)+decompress", WARMUP, ITERS, [&]() {
          auto combined = cuvs::core::multi_and(res, arr[0], 4);
          auto bs       = cuvs::core::to_bitset(res, combined);
        });
      print_result(r_multi);
    }

    printf("\n");
  }

  return 0;
}
