/*
 * Benchmark: Brute-force filtered search — roaring vs bitset
 *
 * Compares filter strategies with interleaved A/B measurement.
 * Brute-force computes full pairwise distances, then masks filtered entries.
 * The filter affects only the masking step (not the GEMM), so differences
 * are expected to be smaller than CAGRA where the filter is in the hot loop.
 *
 * Build:
 *   cd cpp/bench/prims/core/build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j bench_brute_force_roaring
 *   CUDA_CACHE_MAXSIZE=8589934592 \
 *   LD_LIBRARY_PATH=../../../build:../../../build/_deps/rmm-build \
 *       :/mnt/c/Users/maxwb/Development/cu-roaring-filter/build \
 *       ./bench_brute_force_roaring
 */

#include <cuda_runtime.h>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/utils.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/detail/promote.cuh>
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/make_view.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/bitset.cuh>

#include <algorithm>
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
// Stats
// ============================================================================
struct Stats {
  double median, mean, min_v, max_v, std_dev, p5, p95;
  int n;
};

static Stats compute_stats(std::vector<double>& t)
{
  if (t.empty()) return {0, 0, 0, 0, 0, 0, 0, 0};
  std::sort(t.begin(), t.end());
  int n      = static_cast<int>(t.size());
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, t[0], t[n - 1], std::sqrt(var / (n - 1)),
          t[std::max(0, (int)(n * 0.05))],
          t[std::min(n - 1, (int)(n * 0.95))],
          n};
}

static double welch_t(const Stats& a, const Stats& b)
{
  if (a.std_dev == 0 && b.std_dev == 0) return 0;
  double se = std::sqrt((a.std_dev * a.std_dev / a.n) + (b.std_dev * b.std_dev / b.n));
  if (se == 0) return 0;
  return (a.mean - b.mean) / se;
}

static double recall_at_k(const std::vector<int64_t>& result,
                           const std::vector<int64_t>& gt,
                           int k, int n_queries)
{
  int total_found = 0;
  for (int q = 0; q < n_queries; ++q) {
    for (int i = 0; i < k; ++i) {
      int64_t r = result[q * k + i];
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
// Interleaved A/B benchmark
// ============================================================================
struct InterleavedResult {
  Stats bitset, roaring;
  double t_stat;
};

template <typename FilterA, typename FilterB>
InterleavedResult bench_interleaved(
  int warmup, int iters,
  raft::resources& res,
  const cuvs::neighbors::brute_force::search_params& search_params,
  const cuvs::neighbors::brute_force::index<float, float>& index,
  raft::device_matrix_view<const float, int64_t> queries,
  raft::device_matrix_view<int64_t, int64_t> neighbors,
  raft::device_matrix_view<float, int64_t> distances,
  FilterA& filt_a, FilterB& filt_b)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) {
    cuvs::neighbors::brute_force::search(res, search_params, index, queries,
                                          neighbors, distances, filt_a);
    cuvs::neighbors::brute_force::search(res, search_params, index, queries,
                                          neighbors, distances, filt_b);
  }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);

  std::vector<double> times_a(iters), times_b(iters);

  for (int i = 0; i < iters; ++i) {
    auto run = [&](auto& filt) -> double {
      cudaEventRecord(s);
      cuvs::neighbors::brute_force::search(res, search_params, index, queries,
                                            neighbors, distances, filt);
      cudaEventRecord(e);
      cudaEventSynchronize(e);
      float ms;
      cudaEventElapsedTime(&ms, s, e);
      return ms;
    };

    if (i % 2 == 0) {
      times_a[i] = run(filt_a);
      times_b[i] = run(filt_b);
    } else {
      times_b[i] = run(filt_b);
      times_a[i] = run(filt_a);
    }
  }

  cudaEventDestroy(s);
  cudaEventDestroy(e);

  auto sa = compute_stats(times_a);
  auto sb = compute_stats(times_b);
  return {sa, sb, welch_t(sa, sb)};
}

// ============================================================================
// Main benchmark
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

  constexpr int K      = 10;
  constexpr int WARMUP = 10;
  constexpr int ITERS  = 30;

  struct Config {
    const char* name;
    int n_vectors;
    int dim;
    int n_queries;
    double filter_pass_rate;
  };

  Config configs[] = {
    // 100K (dim=128) — small scale, fast iteration
    {"100K_d128_1pct",    100000,  128, 100, 0.01},
    {"100K_d128_5pct",    100000,  128, 100, 0.05},
    {"100K_d128_10pct",   100000,  128, 100, 0.10},
    {"100K_d128_25pct",   100000,  128, 100, 0.25},
    {"100K_d128_50pct",   100000,  128, 100, 0.50},
    // 1M (dim=128) — standard scale
    {"1M_d128_1pct",     1000000,  128, 100, 0.01},
    {"1M_d128_5pct",     1000000,  128, 100, 0.05},
    {"1M_d128_10pct",    1000000,  128, 100, 0.10},
    {"1M_d128_25pct",    1000000,  128, 100, 0.25},
    {"1M_d128_50pct",    1000000,  128, 100, 0.50},
    // 5M (dim=128) — larger scale
    {"5M_d128_1pct",     5000000,  128, 100, 0.01},
    {"5M_d128_5pct",     5000000,  128, 100, 0.05},
    {"5M_d128_10pct",    5000000,  128, 100, 0.10},
    {"5M_d128_25pct",    5000000,  128, 100, 0.25},
    {"5M_d128_50pct",    5000000,  128, 100, 0.50},
    // 10M (dim=128)
    {"10M_d128_1pct",   10000000,  128, 100, 0.01},
    {"10M_d128_10pct",  10000000,  128, 100, 0.10},
    {"10M_d128_50pct",  10000000,  128, 100, 0.50},
    // 20M (dim=128) — dataset ~10GB, tiled GEMM
    {"20M_d128_1pct",   20000000,  128, 100, 0.01},
    {"20M_d128_10pct",  20000000,  128, 100, 0.10},
    {"20M_d128_50pct",  20000000,  128, 100, 0.50},
  };
  constexpr int N_CONFIGS = sizeof(configs) / sizeof(configs[0]);

  FILE* jf = fopen("bench_brute_force_roaring_search.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"brute_force_roaring_search_interleaved\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n  \"n_sms\": %d,\n",
          prop.name, prop.multiProcessorCount);
  fprintf(jf, "  \"l2_cache_mb\": %.0f,\n", prop.l2CacheSize / (1024.0 * 1024.0));
  fprintf(jf, "  \"k\": %d,\n", K);
  fprintf(jf, "  \"metric\": \"L2Expanded\",\n");
  fprintf(jf, "  \"warmup\": %d, \"iters\": %d,\n", WARMUP, ITERS);
  fprintf(jf, "  \"methodology\": \"interleaved_AB\",\n");
  fprintf(jf, "  \"note\": \"bitset uses CSR path for sparsity>=0.9, roaring always uses dense GEMM+mask\",\n");
  fprintf(jf, "  \"results\": [\n");

  bool first_result = true;

  // Reuse index + dataset when N and DIM don't change
  int prev_N = -1, prev_DIM = -1;
  std::unique_ptr<cuvs::neighbors::brute_force::index<float, float>> bf_idx;
  raft::device_matrix<float, int64_t> dataset_buf =
    raft::make_device_matrix<float, int64_t>(res, 0, 0);

  for (int ci = 0; ci < N_CONFIGS; ++ci) {
    auto& cfg = configs[ci];
    int N   = cfg.n_vectors;
    int DIM = cfg.dim;
    int NQ  = cfg.n_queries;

    printf("=== %s (N=%d, dim=%d, NQ=%d, pass=%.1f%%) ===\n",
           cfg.name, N, DIM, NQ, cfg.filter_pass_rate * 100);
    fflush(stdout);

    // Rebuild dataset + index when N or DIM changes
    if (N != prev_N || DIM != prev_DIM) {
      printf("  Generating dataset (%d vectors, %d dim, %.1f MB)...\n",
             N, DIM, (double)N * DIM * sizeof(float) / (1024.0 * 1024.0));
      fflush(stdout);
      dataset_buf = raft::make_device_matrix<float, int64_t>(res, N, DIM);
      {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        std::vector<float> h_data(static_cast<size_t>(N) * DIM);
        for (auto& v : h_data) v = fdist(rng);
        raft::update_device(dataset_buf.data_handle(), h_data.data(), h_data.size(), stream);
        raft::resource::sync_stream(res);
      }

      printf("  Building brute-force index...\n"); fflush(stdout);
      cuvs::neighbors::brute_force::index_params build_params;
      build_params.metric = cuvs::distance::DistanceType::L2Expanded;
      auto idx = cuvs::neighbors::brute_force::build(
        res, build_params, raft::make_const_mdspan(dataset_buf.view()));
      bf_idx = std::make_unique<cuvs::neighbors::brute_force::index<float, float>>(std::move(idx));
      prev_N   = N;
      prev_DIM = DIM;
    }

    // Generate queries
    auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, DIM);
    {
      std::mt19937 rng(99 + ci);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_q(static_cast<size_t>(NQ) * DIM);
      for (auto& v : h_q) v = fdist(rng);
      raft::update_device(queries.data_handle(), h_q.data(), h_q.size(), stream);
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

    double sparsity = 1.0 - static_cast<double>(pass_ids.size()) / N;
    const char* bitset_path = sparsity >= 0.9 ? "CSR+SpGEMM" : "dense+mask";

    // ---- Build filter variants ----

    // Flat bitset
    raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
    {
      uint32_t n_words = (static_cast<uint32_t>(N) + 31) / 32;
      std::vector<uint32_t> h_bits(n_words, 0);
      for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
      raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
    }
    size_t bitset_bytes = (static_cast<size_t>(N) + 31) / 32 * sizeof(uint32_t);

    // Roaring bitmap
    auto gpu_roaring = cu_roaring::upload_from_sorted_ids(
      pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
      static_cast<uint32_t>(N));

    auto roaring_bytes_fn = [](const cu_roaring::GpuRoaring& g) -> size_t {
      return g.n_containers * (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint16_t))
           + static_cast<size_t>(g.n_bitmap_containers) * 1024 * sizeof(uint64_t);
    };
    size_t r_bytes = roaring_bytes_fn(gpu_roaring);

    printf("  Memory: bitset=%.1fKB  roaring=%.1fKB (%.1fx)\n",
           bitset_bytes / 1024.0, r_bytes / 1024.0,
           bitset_bytes / std::max(1.0, (double)r_bytes));
    printf("  Containers: %u (bmp=%u arr=%u)\n",
           gpu_roaring.n_containers,
           gpu_roaring.n_bitmap_containers,
           gpu_roaring.n_array_containers);
    printf("  Bitset path: %s  |  Roaring path: dense+mask\n", bitset_path);

    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, NQ, K);
    auto distances_out = raft::make_device_matrix<float, int64_t>(res, NQ, K);

    cuvs::neighbors::brute_force::search_params search_params;

    // ---- Interleaved benchmark: bitset vs roaring ----
    auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      flat_bitset.view());
    auto roaring_filt = cuvs::neighbors::filtering::roaring_filter(gpu_roaring);

    auto result = bench_interleaved(
      WARMUP, ITERS, res, search_params, *bf_idx,
      raft::make_const_mdspan(queries.view()),
      neighbors.view(), distances_out.view(),
      bitset_filt, roaring_filt);

    auto& s_bitset  = result.bitset;
    auto& s_roaring = result.roaring;

    // Get results for recall calculation
    cuvs::neighbors::brute_force::search(
      res, search_params, *bf_idx,
      raft::make_const_mdspan(queries.view()),
      neighbors.view(), distances_out.view(), bitset_filt);
    std::vector<int64_t> r_bitset(static_cast<size_t>(NQ) * K);
    raft::update_host(r_bitset.data(), neighbors.data_handle(), r_bitset.size(), stream);
    raft::resource::sync_stream(res);

    cuvs::neighbors::brute_force::search(
      res, search_params, *bf_idx,
      raft::make_const_mdspan(queries.view()),
      neighbors.view(), distances_out.view(), roaring_filt);
    std::vector<int64_t> r_roaring(static_cast<size_t>(NQ) * K);
    raft::update_host(r_roaring.data(), neighbors.data_handle(), r_roaring.size(), stream);
    raft::resource::sync_stream(res);

    double recall_roaring = recall_at_k(r_roaring, r_bitset, K, NQ);
    double speedup        = s_bitset.median / s_roaring.median;
    double qps_bitset     = NQ / (s_bitset.median * 1e-3);
    double qps_roaring    = NQ / (s_roaring.median * 1e-3);

    printf("  bitset:  %.3f ms (std=%.3f, p5=%.3f, p95=%.3f) [%s]\n",
           s_bitset.median, s_bitset.std_dev, s_bitset.p5, s_bitset.p95, bitset_path);
    printf("  roaring: %.3f ms (std=%.3f, p5=%.3f, p95=%.3f) [dense+mask]\n",
           s_roaring.median, s_roaring.std_dev, s_roaring.p5, s_roaring.p95);
    printf("  Speedup: %.3fx  Recall: %.4f  t-stat: %.2f\n",
           speedup, recall_roaring, result.t_stat);
    printf("  QPS: bitset=%.0f  roaring=%.0f\n\n", qps_bitset, qps_roaring);

    // ---- JSON ----
    if (!first_result) fprintf(jf, ",\n");
    first_result = false;
    fprintf(jf, "    {\n");
    fprintf(jf, "      \"config\": \"%s\",\n", cfg.name);
    fprintf(jf, "      \"n_vectors\": %d, \"dim\": %d, \"n_queries\": %d,\n", N, DIM, NQ);
    fprintf(jf, "      \"filter_pass_rate\": %.4f, \"n_passing\": %zu,\n",
            cfg.filter_pass_rate, pass_ids.size());
    fprintf(jf, "      \"sparsity\": %.4f,\n", sparsity);
    fprintf(jf, "      \"bitset_path\": \"%s\", \"roaring_path\": \"dense+mask\",\n", bitset_path);
    fprintf(jf, "      \"bitset_bytes\": %zu, \"roaring_bytes\": %zu,\n",
            bitset_bytes, r_bytes);
    fprintf(jf, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u,\n",
            gpu_roaring.n_containers,
            gpu_roaring.n_bitmap_containers,
            gpu_roaring.n_array_containers);
    write_stats_json(jf, "bitset", s_bitset);
    write_stats_json(jf, "roaring", s_roaring);
    fprintf(jf, "      \"recall_roaring\": %.4f,\n", recall_roaring);
    fprintf(jf, "      \"speedup\": %.4f,\n", speedup);
    fprintf(jf, "      \"t_stat\": %.4f,\n", result.t_stat);
    fprintf(jf, "      \"qps_bitset\": %.0f, \"qps_roaring\": %.0f\n",
            qps_bitset, qps_roaring);
    fprintf(jf, "    }");

    cu_roaring::gpu_roaring_free(gpu_roaring);
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  printf("=== COMPLETE — results at bench_brute_force_roaring_search.json ===\n");
  return 0;
}
