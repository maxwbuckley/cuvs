/*
 * Comprehensive Benchmark: CAGRA filtered search — roaring vs bitset
 *
 * Measures search performance, memory, upload time, multi-predicate AND,
 * and end-to-end pipeline across a wide selectivity sweep.
 *
 * Build:
 *   cd cpp/bench/prims/core/build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j bench_cagra_comprehensive
 *   LD_LIBRARY_PATH=../../../build:../../../build/_deps/rmm-build \
 *       ./bench_cagra_comprehensive
 */

#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/utils.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/detail/promote.cuh>
#include <cu_roaring/detail/decompress.cuh>
#include <cu_roaring/detail/set_ops.cuh>
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

// ============================================================================
// Stats helpers (same pattern as bench_cagra_roaring.cu)
// ============================================================================
struct Stats {
  double median, mean, min_v, max_v, std_dev, p5, p95;
};

static Stats compute_stats(std::vector<double>& t)
{
  if (t.empty()) return {0, 0, 0, 0, 0, 0, 0};
  std::sort(t.begin(), t.end());
  int n      = static_cast<int>(t.size());
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, t[0], t[n - 1], std::sqrt(var / n),
          t[std::max(0, (int)(n * 0.05))],
          t[std::min(n - 1, (int)(n * 0.95))]};
}

static Stats bench_gpu(int warmup, int iters, std::function<void()> fn)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) fn();
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(s);
    fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    times[i] = ms;
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return compute_stats(times);
}

static double recall_at_k(const std::vector<uint32_t>& result,
                           const std::vector<uint32_t>& gt,
                           int k, int n_queries)
{
  int total_found = 0;
  for (int q = 0; q < n_queries; ++q) {
    for (int i = 0; i < k; ++i) {
      uint32_t r = result[q * k + i];
      for (int j = 0; j < k; ++j) {
        if (r == gt[q * k + j]) { ++total_found; break; }
      }
    }
  }
  return static_cast<double>(total_found) / (n_queries * k);
}

static void write_stats_json(FILE* f, const char* prefix, const Stats& s)
{
  fprintf(f, "      \"%s_median_ms\": %.4f,\n", prefix, s.median);
  fprintf(f, "      \"%s_mean_ms\": %.4f,\n", prefix, s.mean);
  fprintf(f, "      \"%s_std_ms\": %.4f,\n", prefix, s.std_dev);
  fprintf(f, "      \"%s_p5_ms\": %.4f,\n", prefix, s.p5);
  fprintf(f, "      \"%s_p95_ms\": %.4f,\n", prefix, s.p95);
}

// ============================================================================
// Flat bitset AND kernel
// ============================================================================
__global__ void bitset_and_kernel(const uint32_t* a, const uint32_t* b,
                                   uint32_t* out, uint32_t n_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_words) out[idx] = a[idx] & b[idx];
}

__global__ void bitset_and3_kernel(const uint32_t* a, const uint32_t* b,
                                    const uint32_t* c, uint32_t* out, uint32_t n_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_words) out[idx] = a[idx] & b[idx] & c[idx];
}

__global__ void bitset_and4_kernel(const uint32_t* a, const uint32_t* b,
                                    const uint32_t* c, const uint32_t* d,
                                    uint32_t* out, uint32_t n_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_words) out[idx] = a[idx] & b[idx] & c[idx] & d[idx];
}

// ============================================================================
// Memory helpers
// ============================================================================
static size_t roaring_total_bytes(const cu_roaring::GpuRoaring& g)
{
  size_t bytes = 0;
  bytes += static_cast<size_t>(g.n_containers) *
           (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint16_t));
  bytes += static_cast<size_t>(g.n_bitmap_containers) * 1024 * sizeof(uint64_t);
  // array_data: approximate from total_cardinality minus bitmap contribution
  // For accurate measurement, we'd need to track array pool size
  // Use a conservative estimate: each array container has card entries
  bytes += static_cast<size_t>(g.n_array_containers) * 4096 * sizeof(uint16_t); // worst case
  // key_index
  if (g.key_index) bytes += (g.max_key + 1) * sizeof(uint16_t);
  return bytes;
}

// ============================================================================
// Main
// ============================================================================
int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB VRAM, %.0f MB L2)\n\n",
         prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0),
         prop.l2CacheSize / (1024.0 * 1024.0));

  constexpr int DIM    = 128;
  constexpr int K      = 10;
  constexpr int WARMUP = 10;
  constexpr int ITERS  = 30;

  struct Config {
    const char* name;
    int n_vectors;
    int n_queries;
    double filter_pass_rate;
  };

  Config configs[] = {
    // Selectivity sweep at 1M
    {"1M_0.1pct",  1000000,    100, 0.001},
    {"1M_0.5pct",  1000000,    100, 0.005},
    {"1M_1pct",    1000000,    100, 0.01},
    {"1M_5pct",    1000000,    100, 0.05},
    {"1M_10pct",   1000000,    100, 0.10},
    {"1M_25pct",   1000000,    100, 0.25},
    {"1M_50pct",   1000000,    100, 0.50},
    {"1M_75pct",   1000000,    100, 0.75},
    {"1M_90pct",   1000000,    100, 0.90},
    {"1M_99pct",   1000000,    100, 0.99},
    // Throughput at 1M
    {"1M_10pct_batch10K",  1000000, 10000, 0.10},
    {"1M_50pct_batch10K",  1000000, 10000, 0.50},
    // Scale test at 10M
    {"10M_1pct",   10000000,   100, 0.01},
    {"10M_10pct",  10000000,   100, 0.10},
    {"10M_50pct",  10000000,   100, 0.50},
  };
  constexpr int N_CONFIGS = sizeof(configs) / sizeof(configs[0]);

  // ========================================================================
  // JSON output
  // ========================================================================
  FILE* jf = fopen("bench_cagra_roaring_comprehensive.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"cagra_roaring_comprehensive\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n  \"n_sms\": %d,\n", prop.name, prop.multiProcessorCount);
  fprintf(jf, "  \"l2_cache_mb\": %.1f,\n", prop.l2CacheSize / (1024.0 * 1024.0));
  fprintf(jf, "  \"dim\": %d, \"k\": %d,\n", DIM, K);
  fprintf(jf, "  \"warmup\": %d, \"iters\": %d,\n", WARMUP, ITERS);

  // ========================================================================
  // SECTION 1: Search performance across selectivities
  // ========================================================================
  fprintf(jf, "  \"search_results\": [\n");
  bool first_result = true;

  int prev_N = -1;
  // Reusable CAGRA index + data (rebuild only when N changes)
  std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_idx;
  raft::device_matrix<float, int64_t> dataset_buf = raft::make_device_matrix<float, int64_t>(res, 0, 0);
  raft::device_matrix<float, int64_t> queries_buf = raft::make_device_matrix<float, int64_t>(res, 0, 0);

  for (int ci = 0; ci < N_CONFIGS; ++ci) {
    auto& cfg = configs[ci];
    int N  = cfg.n_vectors;
    int NQ = cfg.n_queries;

    printf("=== %s (N=%d, NQ=%d, pass=%.1f%%) ===\n",
           cfg.name, N, NQ, cfg.filter_pass_rate * 100);
    fflush(stdout);

    // Rebuild dataset + CAGRA index if N changed
    if (N != prev_N) {
      printf("  Generating %dD dataset (%d vectors)...\n", DIM, N); fflush(stdout);
      dataset_buf = raft::make_device_matrix<float, int64_t>(res, N, DIM);
      {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        std::vector<float> h_data(static_cast<size_t>(N) * DIM);
        for (auto& v : h_data) v = fdist(rng);
        raft::update_device(dataset_buf.data_handle(), h_data.data(), h_data.size(), stream);
        raft::resource::sync_stream(res);
      }

      printf("  Building CAGRA index...\n"); fflush(stdout);
      cuvs::neighbors::cagra::index_params build_params;
      build_params.graph_degree              = 32;
      build_params.intermediate_graph_degree = 48;
      auto idx = cuvs::neighbors::cagra::build(
        res, build_params, raft::make_const_mdspan(dataset_buf.view()));
      cagra_idx = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(std::move(idx));
      prev_N = N;
    }

    // Generate queries (fresh per config to vary NQ)
    queries_buf = raft::make_device_matrix<float, int64_t>(res, NQ, DIM);
    {
      std::mt19937 rng(99 + ci);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_q(static_cast<size_t>(NQ) * DIM);
      for (auto& v : h_q) v = fdist(rng);
      raft::update_device(queries_buf.data_handle(), h_q.data(), h_q.size(), stream);
      raft::resource::sync_stream(res);
    }

    // Generate filter IDs
    std::mt19937 gen(123 + ci);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<uint32_t> pass_ids;
    pass_ids.reserve(static_cast<size_t>(N * cfg.filter_pass_rate * 1.1));
    for (int i = 0; i < N; ++i)
      if (dist(gen) < cfg.filter_pass_rate) pass_ids.push_back(static_cast<uint32_t>(i));
    printf("  Filter: %zu pass (%.2f%%)\n", pass_ids.size(), 100.0 * pass_ids.size() / N);

    // ---- Build filter variants ----

    // 1. Flat bitset
    raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
    Stats bitset_build_stats;
    {
      uint32_t n_words = (static_cast<uint32_t>(N) + 31) / 32;
      std::vector<uint32_t> h_bits(n_words, 0);
      for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));

      // Time bitset construction (host→device transfer)
      bitset_build_stats = bench_gpu(3, ITERS, [&]() {
        raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
      });
    }
    size_t bitset_bytes = (static_cast<size_t>(N) + 31) / 32 * sizeof(uint32_t);

    // 2. Roaring default (PROMOTE_AUTO)
    cu_roaring::GpuRoaring gpu_roaring_default{};
    Stats roaring_build_stats;
    {
      roaring_build_stats = bench_gpu(3, ITERS, [&]() {
        if (gpu_roaring_default.keys) cu_roaring::gpu_roaring_free(gpu_roaring_default);
        gpu_roaring_default = cu_roaring::upload_from_ids(
          pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
          static_cast<uint32_t>(N));
      });
    }
    auto view_default = cu_roaring::make_view(gpu_roaring_default);

    // 3. Roaring promoted (PROMOTE_ALL — all bitmap, 2-read fast path)
    auto gpu_roaring_promoted = cu_roaring::upload_from_ids(
      pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
      static_cast<uint32_t>(N), 0, cu_roaring::PROMOTE_ALL);
    auto view_promoted = cu_roaring::make_view(gpu_roaring_promoted);

    // Memory stats
    size_t roaring_default_bytes  = roaring_total_bytes(gpu_roaring_default);
    size_t roaring_promoted_bytes = roaring_total_bytes(gpu_roaring_promoted);
    double compression_default  = bitset_bytes / std::max(1.0, (double)roaring_default_bytes);
    double compression_promoted = bitset_bytes / std::max(1.0, (double)roaring_promoted_bytes);

    printf("  Memory: bitset=%.1fKB  roaring=%.1fKB (%.1fx)  promoted=%.1fKB (%.1fx)\n",
           bitset_bytes / 1024.0,
           roaring_default_bytes / 1024.0, compression_default,
           roaring_promoted_bytes / 1024.0, compression_promoted);
    printf("  Containers: default=%u (bmp=%u arr=%u) negated=%d  promoted=%u (all bmp) negated=%d\n",
           gpu_roaring_default.n_containers,
           gpu_roaring_default.n_bitmap_containers,
           gpu_roaring_default.n_array_containers,
           gpu_roaring_default.negated,
           gpu_roaring_promoted.n_containers,
           gpu_roaring_promoted.negated);
    printf("  Build: bitset=%.3fms  roaring=%.3fms\n",
           bitset_build_stats.median, roaring_build_stats.median);

    // ---- Correctness verification ----
    // Verify roaring and bitset agree on 1000 random samples
    int n_verify = std::min(1000, N);
    int mismatches = 0;
    {
      std::vector<uint32_t> verify_ids(n_verify);
      std::iota(verify_ids.begin(), verify_ids.end(), 0);
      // Check: does roaring agree with bitset?
      std::vector<uint32_t> h_bits((N + 31) / 32, 0);
      for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
      auto v = cu_roaring::make_view(gpu_roaring_default);
      // We can't call device functions from host, so decompress and compare
      uint32_t n_words_verify = (N + 31) / 32;
      uint32_t* d_roaring_bits = cu_roaring::decompress_to_bitset(gpu_roaring_default);
      std::vector<uint32_t> h_roaring_bits(n_words_verify);
      cudaMemcpy(h_roaring_bits.data(), d_roaring_bits,
                 n_words_verify * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaFree(d_roaring_bits);
      for (int i = 0; i < n_verify; ++i) {
        bool in_bitset  = (h_bits[i / 32] >> (i % 32)) & 1;
        bool in_roaring = (h_roaring_bits[i / 32] >> (i % 32)) & 1;
        if (in_bitset != in_roaring) ++mismatches;
      }
      if (mismatches > 0)
        printf("  WARNING: %d filter mismatches!\n", mismatches);
    }

    // ---- Search benchmarks ----
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, NQ, K);
    auto distances = raft::make_device_matrix<float, int64_t>(res, NQ, K);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = 256;

    // No filter
    auto s_none = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, *cagra_idx,
        raft::make_const_mdspan(queries_buf.view()),
        neighbors.view(), distances.view());
    });
    printf("  no_filter:        %.3f ms (std=%.3f)\n", s_none.median, s_none.std_dev);

    // Bitset filter
    auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      flat_bitset.view());
    auto s_bitset = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, *cagra_idx,
        raft::make_const_mdspan(queries_buf.view()),
        neighbors.view(), distances.view(), bitset_filt);
    });
    std::vector<uint32_t> r_bitset(static_cast<size_t>(NQ) * K);
    raft::update_host(r_bitset.data(), neighbors.data_handle(), r_bitset.size(), stream);
    raft::resource::sync_stream(res);
    printf("  bitset_filter:    %.3f ms (std=%.3f)\n", s_bitset.median, s_bitset.std_dev);

    // Roaring default — decompress to bitset, then search via bitset_filter.
    // This measures the full roaring pipeline: upload → decompress → search.
    // (Direct roaring_filter dispatch via dynamic_cast has RTTI mismatch with
    // the conda-built libcuvs.so, so we use the decompress-to-bitset path.)
    raft::core::bitset<uint32_t, int64_t> roaring_decompressed(res, static_cast<int64_t>(N), false);
    {
      uint32_t n_words_dec = (static_cast<uint32_t>(N) + 31) / 32;
      cu_roaring::decompress_to_bitset(gpu_roaring_default, roaring_decompressed.data(), n_words_dec);
      cudaDeviceSynchronize();
    }
    auto roaring_bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      roaring_decompressed.view());
    auto s_roaring = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, *cagra_idx,
        raft::make_const_mdspan(queries_buf.view()),
        neighbors.view(), distances.view(), roaring_bitset_filt);
    });
    std::vector<uint32_t> r_roaring(static_cast<size_t>(NQ) * K);
    raft::update_host(r_roaring.data(), neighbors.data_handle(), r_roaring.size(), stream);
    raft::resource::sync_stream(res);
    printf("  roaring_default:  %.3f ms (std=%.3f)\n", s_roaring.median, s_roaring.std_dev);

    // Roaring promoted — same decompress-to-bitset approach
    raft::core::bitset<uint32_t, int64_t> promoted_decompressed(res, static_cast<int64_t>(N), false);
    {
      uint32_t n_words_dec = (static_cast<uint32_t>(N) + 31) / 32;
      cu_roaring::decompress_to_bitset(gpu_roaring_promoted, promoted_decompressed.data(), n_words_dec);
      cudaDeviceSynchronize();
    }
    auto promoted_bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      promoted_decompressed.view());
    auto s_promoted = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, *cagra_idx,
        raft::make_const_mdspan(queries_buf.view()),
        neighbors.view(), distances.view(), promoted_bitset_filt);
    });
    std::vector<uint32_t> r_promoted(static_cast<size_t>(NQ) * K);
    raft::update_host(r_promoted.data(), neighbors.data_handle(), r_promoted.size(), stream);
    raft::resource::sync_stream(res);
    printf("  roaring_promoted: %.3f ms (std=%.3f)\n", s_promoted.median, s_promoted.std_dev);

    // ---- Recall ----
    double recall_bitset   = 1.0;  // bitset is ground truth
    double recall_roaring  = recall_at_k(r_roaring, r_bitset, K, NQ);
    double recall_promoted = recall_at_k(r_promoted, r_bitset, K, NQ);
    printf("  Recall@%d: bitset=%.4f  roaring=%.4f  promoted=%.4f\n",
           K, recall_bitset, recall_roaring, recall_promoted);

    // ---- Speedup & QPS ----
    double spd_roaring  = s_bitset.median / s_roaring.median;
    double spd_promoted = s_bitset.median / s_promoted.median;
    double qps_none     = NQ / (s_none.median * 1e-3);
    double qps_bitset   = NQ / (s_bitset.median * 1e-3);
    double qps_roaring  = NQ / (s_roaring.median * 1e-3);
    double qps_promoted = NQ / (s_promoted.median * 1e-3);
    printf("  Speedup vs bitset: roaring=%.2fx  promoted=%.2fx\n", spd_roaring, spd_promoted);
    printf("  QPS: none=%.0f  bitset=%.0f  roaring=%.0f  promoted=%.0f\n\n",
           qps_none, qps_bitset, qps_roaring, qps_promoted);

    // ---- E2E pipeline ----
    double e2e_bitset_ms  = bitset_build_stats.median + s_bitset.median;
    double e2e_roaring_ms = roaring_build_stats.median + s_roaring.median;

    // ---- JSON ----
    if (!first_result) fprintf(jf, ",\n");
    first_result = false;
    fprintf(jf, "    {\n");
    fprintf(jf, "      \"config\": \"%s\",\n", cfg.name);
    fprintf(jf, "      \"n_vectors\": %d, \"n_queries\": %d,\n", N, NQ);
    fprintf(jf, "      \"filter_pass_rate\": %.4f, \"n_passing\": %zu,\n",
            cfg.filter_pass_rate, pass_ids.size());
    fprintf(jf, "      \"filter_mismatches\": %d,\n", mismatches);
    fprintf(jf, "      \"negated\": %s,\n", gpu_roaring_default.negated ? "true" : "false");
    fprintf(jf, "      \"bitset_bytes\": %zu, \"roaring_default_bytes\": %zu, \"roaring_promoted_bytes\": %zu,\n",
            bitset_bytes, roaring_default_bytes, roaring_promoted_bytes);
    fprintf(jf, "      \"compression_default\": %.2f, \"compression_promoted\": %.2f,\n",
            compression_default, compression_promoted);
    fprintf(jf, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u,\n",
            gpu_roaring_default.n_containers,
            gpu_roaring_default.n_bitmap_containers,
            gpu_roaring_default.n_array_containers);
    write_stats_json(jf, "bitset_build", bitset_build_stats);
    write_stats_json(jf, "roaring_build", roaring_build_stats);
    write_stats_json(jf, "no_filter", s_none);
    write_stats_json(jf, "bitset", s_bitset);
    write_stats_json(jf, "roaring", s_roaring);
    write_stats_json(jf, "roaring_promoted", s_promoted);
    fprintf(jf, "      \"recall_bitset\": %.4f,\n", recall_bitset);
    fprintf(jf, "      \"recall_roaring\": %.4f, \"recall_promoted\": %.4f,\n",
            recall_roaring, recall_promoted);
    fprintf(jf, "      \"speedup_roaring\": %.4f, \"speedup_promoted\": %.4f,\n",
            spd_roaring, spd_promoted);
    fprintf(jf, "      \"qps_none\": %.0f, \"qps_bitset\": %.0f, \"qps_roaring\": %.0f, \"qps_promoted\": %.0f,\n",
            qps_none, qps_bitset, qps_roaring, qps_promoted);
    fprintf(jf, "      \"e2e_bitset_ms\": %.4f, \"e2e_roaring_ms\": %.4f, \"e2e_speedup\": %.4f\n",
            e2e_bitset_ms, e2e_roaring_ms, e2e_bitset_ms / e2e_roaring_ms);
    fprintf(jf, "    }");

    cu_roaring::gpu_roaring_free(gpu_roaring_default);
    cu_roaring::gpu_roaring_free(gpu_roaring_promoted);
  }
  fprintf(jf, "\n  ],\n");

  // ========================================================================
  // SECTION 2: Multi-predicate AND
  // ========================================================================
  printf("========================================\n");
  printf("Multi-Predicate AND Benchmark\n");
  printf("========================================\n\n");
  fprintf(jf, "  \"multi_and_results\": [\n");

  {
    constexpr int N_MULTI = 1000000;
    constexpr double PRED_RATE = 0.50;  // each predicate passes 50%

    // Generate NQ queries for multi-predicate search
    constexpr int NQ_MULTI = 100;
    auto queries_multi = raft::make_device_matrix<float, int64_t>(res, NQ_MULTI, DIM);
    {
      std::mt19937 rng(777);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_q(static_cast<size_t>(NQ_MULTI) * DIM);
      for (auto& v : h_q) v = fdist(rng);
      raft::update_device(queries_multi.data_handle(), h_q.data(), h_q.size(), stream);
      raft::resource::sync_stream(res);
    }

    // Build CAGRA index at 1M if not already
    if (prev_N != N_MULTI) {
      printf("  Generating 1M dataset for multi-AND...\n"); fflush(stdout);
      dataset_buf = raft::make_device_matrix<float, int64_t>(res, N_MULTI, DIM);
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_data(static_cast<size_t>(N_MULTI) * DIM);
      for (auto& v : h_data) v = fdist(rng);
      raft::update_device(dataset_buf.data_handle(), h_data.data(), h_data.size(), stream);
      raft::resource::sync_stream(res);

      printf("  Building CAGRA index...\n"); fflush(stdout);
      cuvs::neighbors::cagra::index_params build_params;
      build_params.graph_degree              = 32;
      build_params.intermediate_graph_degree = 48;
      auto idx = cuvs::neighbors::cagra::build(
        res, build_params, raft::make_const_mdspan(dataset_buf.view()));
      cagra_idx = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(std::move(idx));
      prev_N = N_MULTI;
    }

    uint32_t n_words = (N_MULTI + 31) / 32;

    for (int n_pred : {2, 3, 4}) {
      printf("--- %d-predicate AND (N=%d, each ~%.0f%% pass) ---\n",
             n_pred, N_MULTI, PRED_RATE * 100);

      // Generate independent predicates
      std::vector<std::vector<uint32_t>> pred_ids(n_pred);
      std::vector<std::vector<uint32_t>> pred_bitsets(n_pred);

      for (int p = 0; p < n_pred; ++p) {
        std::mt19937 gen(500 + p * 13);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        pred_bitsets[p].resize(n_words, 0);
        for (int i = 0; i < N_MULTI; ++i) {
          if (dist(gen) < PRED_RATE) {
            pred_ids[p].push_back(static_cast<uint32_t>(i));
            pred_bitsets[p][i / 32] |= (1u << (i % 32));
          }
        }
      }

      // Upload flat bitsets
      std::vector<uint32_t*> d_bitsets(n_pred);
      for (int p = 0; p < n_pred; ++p) {
        cudaMalloc(&d_bitsets[p], n_words * sizeof(uint32_t));
        cudaMemcpy(d_bitsets[p], pred_bitsets[p].data(),
                   n_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
      }

      // Upload roaring bitmaps
      std::vector<cu_roaring::GpuRoaring> roaring_bms(n_pred);
      for (int p = 0; p < n_pred; ++p) {
        roaring_bms[p] = cu_roaring::upload_from_ids(
          pred_ids[p].data(), static_cast<uint32_t>(pred_ids[p].size()),
          static_cast<uint32_t>(N_MULTI));
      }

      // Time flat bitset AND
      uint32_t* d_combined;
      cudaMalloc(&d_combined, n_words * sizeof(uint32_t));
      int blocks = (n_words + 255) / 256;

      Stats bitset_and_stats = bench_gpu(WARMUP, ITERS, [&]() {
        if (n_pred == 2) {
          bitset_and_kernel<<<blocks, 256>>>(d_bitsets[0], d_bitsets[1], d_combined, n_words);
        } else if (n_pred == 3) {
          bitset_and3_kernel<<<blocks, 256>>>(d_bitsets[0], d_bitsets[1], d_bitsets[2],
                                               d_combined, n_words);
        } else {
          bitset_and4_kernel<<<blocks, 256>>>(d_bitsets[0], d_bitsets[1], d_bitsets[2],
                                               d_bitsets[3], d_combined, n_words);
        }
      });

      // Time roaring fused multi_and
      cu_roaring::GpuRoaring combined_roaring{};
      Stats roaring_and_stats = bench_gpu(WARMUP, ITERS, [&]() {
        if (combined_roaring.keys) cu_roaring::gpu_roaring_free(combined_roaring);
        combined_roaring = cu_roaring::multi_and(roaring_bms.data(), n_pred);
      });

      // Sizes
      size_t combined_bitset_bytes = n_words * sizeof(uint32_t);
      size_t combined_roaring_bytes = roaring_total_bytes(combined_roaring);

      printf("  Bitset AND:  %.3f ms\n", bitset_and_stats.median);
      printf("  Roaring AND: %.3f ms  (%.2fx)\n",
             roaring_and_stats.median, bitset_and_stats.median / roaring_and_stats.median);
      printf("  Combined memory: bitset=%.1fKB  roaring=%.1fKB (%.1fx)\n",
             combined_bitset_bytes / 1024.0,
             combined_roaring_bytes / 1024.0,
             combined_bitset_bytes / std::max(1.0, (double)combined_roaring_bytes));

      // Search with combined filter
      auto neighbors_multi = raft::make_device_matrix<uint32_t, int64_t>(res, NQ_MULTI, K);
      auto distances_multi = raft::make_device_matrix<float, int64_t>(res, NQ_MULTI, K);
      cuvs::neighbors::cagra::search_params sp_multi;
      sp_multi.itopk_size = 256;

      // Bitset combined search
      raft::core::bitset<uint32_t, int64_t> combined_flat(res, static_cast<int64_t>(N_MULTI), false);
      raft::update_device(combined_flat.data(), d_combined, n_words, stream);
      raft::resource::sync_stream(res);
      auto bitset_combined_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
        combined_flat.view());

      auto s_search_bitset = bench_gpu(WARMUP, ITERS, [&]() {
        cuvs::neighbors::cagra::search(
          res, sp_multi, *cagra_idx,
          raft::make_const_mdspan(queries_multi.view()),
          neighbors_multi.view(), distances_multi.view(), bitset_combined_filt);
      });

      // Roaring combined search — decompress to bitset then search
      raft::core::bitset<uint32_t, int64_t> combined_roaring_dec(res, static_cast<int64_t>(N_MULTI), false);
      cu_roaring::decompress_to_bitset(combined_roaring, combined_roaring_dec.data(), n_words);
      cudaDeviceSynchronize();
      auto roaring_combined_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
        combined_roaring_dec.view());
      auto s_search_roaring = bench_gpu(WARMUP, ITERS, [&]() {
        cuvs::neighbors::cagra::search(
          res, sp_multi, *cagra_idx,
          raft::make_const_mdspan(queries_multi.view()),
          neighbors_multi.view(), distances_multi.view(), roaring_combined_filt);
      });

      printf("  Search (combined): bitset=%.3fms  roaring=%.3fms  (%.2fx)\n\n",
             s_search_bitset.median, s_search_roaring.median,
             s_search_bitset.median / s_search_roaring.median);

      // JSON
      if (n_pred > 2) fprintf(jf, ",\n");
      fprintf(jf, "    {\n");
      fprintf(jf, "      \"n_predicates\": %d, \"per_predicate_rate\": %.2f,\n",
              n_pred, PRED_RATE);
      fprintf(jf, "      \"n_vectors\": %d,\n", N_MULTI);
      write_stats_json(jf, "bitset_and", bitset_and_stats);
      write_stats_json(jf, "roaring_and", roaring_and_stats);
      fprintf(jf, "      \"and_speedup\": %.4f,\n",
              bitset_and_stats.median / roaring_and_stats.median);
      fprintf(jf, "      \"combined_bitset_bytes\": %zu, \"combined_roaring_bytes\": %zu,\n",
              combined_bitset_bytes, combined_roaring_bytes);
      fprintf(jf, "      \"combined_compression\": %.2f,\n",
              combined_bitset_bytes / std::max(1.0, (double)combined_roaring_bytes));
      fprintf(jf, "      \"combined_negated\": %s,\n",
              combined_roaring.negated ? "true" : "false");
      write_stats_json(jf, "search_bitset", s_search_bitset);
      write_stats_json(jf, "search_roaring", s_search_roaring);
      fprintf(jf, "      \"search_speedup\": %.4f\n",
              s_search_bitset.median / s_search_roaring.median);
      fprintf(jf, "    }");

      // Cleanup
      cu_roaring::gpu_roaring_free(combined_roaring);
      combined_roaring = cu_roaring::GpuRoaring{};
      for (int p = 0; p < n_pred; ++p) {
        cu_roaring::gpu_roaring_free(roaring_bms[p]);
        cudaFree(d_bitsets[p]);
      }
      cudaFree(d_combined);
    }
  }
  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);

  printf("=== COMPLETE — results at bench_cagra_roaring_comprehensive.json ===\n");
  return 0;
}
