/*
 * Benchmark: transfer + search latency comparison.
 *
 * Measures the end-to-end cost of:
 *   "I have my filter ready on the CPU — how fast can I search?"
 *
 * Bitset path:  cudaMemcpy(bitset) → bitset_filter → search
 * Roaring path: cu_roaring::upload(croaring) → roaring_filter → search
 *
 * Excludes: CPU filter construction, CAGRA index build.
 * Includes: host→device transfer, filter object creation, search kernel.
 */

#include <cuda_runtime.h>
#include <roaring/roaring.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/upload.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/upload_pool.hpp>
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/make_view.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/bitset.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

struct Stats {
  double median, mean, std_dev;
  int n;
};

static Stats compute_stats(std::vector<double>& t)
{
  std::sort(t.begin(), t.end());
  int n      = static_cast<int>(t.size());
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, std::sqrt(var / (n - 1)), n};
}

static double welch_t(const Stats& a, const Stats& b)
{
  double se = std::sqrt((a.std_dev * a.std_dev / a.n) + (b.std_dev * b.std_dev / b.n));
  if (se == 0) return 0;
  return (a.mean - b.mean) / se;
}

int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n\n", prop.name);

  constexpr int K      = 10;
  constexpr int WARMUP = 10;
  constexpr int ITERS  = 30;

  struct Config {
    const char* name;
    int n_vectors, dim, n_queries;
    double pass_rate;
  };

  Config configs[] = {
    // 1M (dim=2) — minimal vectors, maximize filter fraction of total time
    {"1M_d2_0.1pct",   1000000, 2, 100, 0.001},
    {"1M_d2_1pct",     1000000, 2, 100, 0.01},
    {"1M_d2_10pct",    1000000, 2, 100, 0.10},
    {"1M_d2_50pct",    1000000, 2, 100, 0.50},
    // 10M (dim=2) — bitset = 1.2MB, croaring at 0.1% = ~20KB
    {"10M_d2_0.1pct", 10000000, 2, 100, 0.001},
    {"10M_d2_1pct",   10000000, 2, 100, 0.01},
    {"10M_d2_10pct",  10000000, 2, 100, 0.10},
    {"10M_d2_50pct",  10000000, 2, 100, 0.50},
    // 20M (dim=2) — bitset = 2.4MB, croaring at 0.1% = ~40KB
    {"20M_d2_0.1pct", 20000000, 2, 100, 0.001},
    {"20M_d2_1pct",   20000000, 2, 100, 0.01},
    // 40M (dim=2) — bitset = 4.9MB, croaring at 0.1% = ~80KB
    {"40M_d2_0.1pct", 40000000, 2, 100, 0.001},
    {"40M_d2_1pct",   40000000, 2, 100, 0.01},
    {"40M_d2_10pct",  40000000, 2, 100, 0.10},
  };

  // Reuse CAGRA index across configs (same N, DIM)
  int prev_N = -1, prev_DIM = -1;
  std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_idx;
  raft::device_matrix<float, int64_t> dataset_buf =
    raft::make_device_matrix<float, int64_t>(res, 0, 0);

  printf("%-20s | %7s %5s | %7s %5s | %7s %5s | %5s %5s | %s\n",
         "Config", "Bitset", "std", "Roar", "std", "Pool", "std",
         "R/B", "P/B", "GPU(KB)");
  printf("%-20s-+-%7s-%5s-+-%7s-%5s-+-%7s-%5s-+-%5s-%5s-+-%s\n",
         "--------------------", "-------", "-----", "-------", "-----",
         "-------", "-----", "-----", "-----", "-------");

  for (auto& cfg : configs) {
    int N = cfg.n_vectors, DIM = cfg.dim, NQ = cfg.n_queries;

    if (N != prev_N || DIM != prev_DIM) {
      fprintf(stderr, "Building dataset + CAGRA index (%d, %d)...\n", N, DIM);
      dataset_buf = raft::make_device_matrix<float, int64_t>(res, N, DIM);
      {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        std::vector<float> h(static_cast<size_t>(N) * DIM);
        for (auto& v : h) v = fdist(rng);
        raft::update_device(dataset_buf.data_handle(), h.data(), h.size(), stream);
        raft::resource::sync_stream(res);
      }
      cuvs::neighbors::cagra::index_params bp;
      bp.graph_degree = 32;
      bp.intermediate_graph_degree = 48;
      auto idx = cuvs::neighbors::cagra::build(
        res, bp, raft::make_const_mdspan(dataset_buf.view()));
      cagra_idx = std::make_unique<decltype(idx)>(std::move(idx));
      prev_N = N; prev_DIM = DIM;
    }

    // Queries (device)
    auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, DIM);
    {
      std::mt19937 rng(99);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h(static_cast<size_t>(NQ) * DIM);
      for (auto& v : h) v = fdist(rng);
      raft::update_device(queries.data_handle(), h.data(), h.size(), stream);
      raft::resource::sync_stream(res);
    }
    auto q_view = raft::make_const_mdspan(queries.view());

    // Output buffers (device, reused)
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, NQ, K);
    auto distances = raft::make_device_matrix<float, int64_t>(res, NQ, K);

    // Search params with filtering_rate pre-set (no popcount overhead)
    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size    = 256;
    sp.filtering_rate = static_cast<float>(1.0 - cfg.pass_rate);

    // ---- Prepare CPU-side filter representations (NOT timed) ----

    // Generate passing IDs
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<uint32_t> pass_ids;
    for (int i = 0; i < N; ++i)
      if (dist(gen) < cfg.pass_rate) pass_ids.push_back(static_cast<uint32_t>(i));

    // CPU bitset (host memory)
    uint32_t n_words = (static_cast<uint32_t>(N) + 31) / 32;
    std::vector<uint32_t> h_bitset(n_words, 0);
    for (auto id : pass_ids) h_bitset[id / 32] |= (1u << (id % 32));
    size_t bitset_bytes = n_words * sizeof(uint32_t);

    // CPU CRoaring bitmap
    roaring_bitmap_t* croaring = roaring_bitmap_create();
    for (auto id : pass_ids) roaring_bitmap_add(croaring, id);
    size_t croaring_bytes = roaring_bitmap_size_in_bytes(croaring);

    // Compute GPU-side roaring size with PROMOTE_NONE
    auto meta = cu_roaring::get_meta(croaring);
    size_t roaring_gpu_bytes = meta.total_bytes;

    fprintf(stderr, "  %s: %zu pass, bitset=%zuKB, croaring_cpu=%zuKB, roaring_gpu=%zuKB (%.1fx smaller)\n",
            cfg.name, pass_ids.size(), bitset_bytes / 1024,
            croaring_bytes / 1024, roaring_gpu_bytes / 1024,
            static_cast<double>(bitset_bytes) / std::max(size_t(1), roaring_gpu_bytes));

    // ---- Timed region: transfer + filter + search ----

    // (wall-clock timing used instead of GPU events — see run_bitset/run_roaring)

    // Pre-allocate device bitset (reused across iterations to match real usage)
    raft::core::bitset<uint32_t, int64_t> dev_bitset(res, static_cast<int64_t>(N), false);

    // Wall-clock timing (not GPU events) because cu_roaring::upload() does
    // host-side work (parsing, cudaMalloc) that GPU events wouldn't capture.
    auto wall_ms = [](auto fn) -> double {
      auto t0 = std::chrono::high_resolution_clock::now();
      fn();
      auto t1 = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    auto run_bitset = [&]() -> double {
      return wall_ms([&] {
        // Transfer bitset host → device
        raft::update_device(dev_bitset.data(), h_bitset.data(), n_words, stream);
        auto filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
          dev_bitset.view());
        // Search
        cuvs::neighbors::cagra::search(res, sp, *cagra_idx, q_view,
                                        neighbors.view(), distances.view(), filt);
        cudaDeviceSynchronize();
      });
    };

    auto run_roaring = [&]() -> double {
      return wall_ms([&] {
        // Upload compressed CRoaring → GPU, NO promotion (keep array containers)
        auto gpu = cu_roaring::upload(croaring, static_cast<uint32_t>(N), stream,
                                      cu_roaring::PROMOTE_NONE);
        auto filt = cuvs::neighbors::filtering::roaring_filter(gpu);
        // Search
        cuvs::neighbors::cagra::search(res, sp, *cagra_idx, q_view,
                                        neighbors.view(), distances.view(), filt);
        cudaDeviceSynchronize();
        cu_roaring::gpu_roaring_free(gpu);
      });
    };

    // Pool: pre-allocated buffers, zero malloc per call
    cu_roaring::UploadPool pool(8 * 1024 * 1024);  // 8 MB pool

    auto run_pool = [&]() -> double {
      return wall_ms([&] {
        auto gpu = pool.upload(croaring, static_cast<uint32_t>(N), stream,
                               cu_roaring::PROMOTE_NONE);
        auto filt = cuvs::neighbors::filtering::roaring_filter(gpu);
        cuvs::neighbors::cagra::search(res, sp, *cagra_idx, q_view,
                                        neighbors.view(), distances.view(), filt);
        cudaDeviceSynchronize();
        // No gpu_roaring_free — pool owns the memory
      });
    };

    // Warmup (all three, interleaved)
    for (int i = 0; i < WARMUP; ++i) { run_bitset(); run_roaring(); run_pool(); }

    // Measured (interleaved)
    std::vector<double> t_bitset(ITERS), t_roaring(ITERS), t_pool(ITERS);
    for (int i = 0; i < ITERS; ++i) {
      if (i % 2 == 0) {
        t_bitset[i]  = run_bitset();
        t_roaring[i] = run_roaring();
        t_pool[i]    = run_pool();
      } else {
        t_pool[i]    = run_pool();
        t_roaring[i] = run_roaring();
        t_bitset[i]  = run_bitset();
      }
    }

    auto sb = compute_stats(t_bitset);
    auto sr = compute_stats(t_roaring);
    auto sp2 = compute_stats(t_pool);
    double t_br = welch_t(sb, sr);
    double t_bp = welch_t(sb, sp2);

    printf("%-20s | %7.3f %5.3f | %7.3f %5.3f | %7.3f %5.3f | %5.2fx %5.2fx | %4zu/%zu\n",
           cfg.name,
           sb.median, sb.std_dev,
           sr.median, sr.std_dev,
           sp2.median, sp2.std_dev,
           sb.median / sr.median,
           sb.median / sp2.median,
           roaring_gpu_bytes / 1024, bitset_bytes / 1024);
    fflush(stdout);

    roaring_bitmap_free(croaring);
  }

  printf("\nXfer = CRoaring serialized size / bitset size (KB)\n");
  return 0;
}
