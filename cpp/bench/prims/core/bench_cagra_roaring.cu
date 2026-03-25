/*
 * Benchmark: CAGRA filtered search — roaring vs bitset
 *
 * Compares four filter strategies across selectivities and dataset sizes:
 *   1. bitset_filter    — cuVS native flat bitset (baseline)
 *   2. roaring_warp     — compressed Roaring, warp-cooperative (CRoaring threshold)
 *   3. roaring_promoted — compressed Roaring, all containers promoted to bitmap
 *   4. no_filter        — unfiltered CAGRA search (reference)
 *
 * Build:
 *   cd cpp/bench/prims/core/build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j
 *   LD_LIBRARY_PATH=../../../build:../../../build/_deps/rmm-build ./bench_cagra_roaring
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
void build_key_bloom(GpuRoaring& bitmap, cudaStream_t stream);
void gpu_roaring_free(GpuRoaring& bitmap);
}

// ============================================================================
// Helpers
// ============================================================================
struct Stats {
  double median, mean, min_v, max_v, std_dev, p5, p95;
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
// Main benchmark
// ============================================================================
int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB)\n\n", prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0));

  constexpr int DIM       = 128;
  constexpr int K         = 10;
  constexpr int WARMUP    = 10;
  constexpr int ITERS     = 30;

  struct Config {
    const char* name;
    int n_vectors;
    int n_queries;
    double filter_pass_rate;
  };

  Config configs[] = {
    // 1M dataset — various selectivities
    {"1M_50pct",  1000000, 100, 0.50},
    {"1M_10pct",  1000000, 100, 0.10},
    {"1M_1pct",   1000000, 100, 0.01},
    // 1M throughput — large batch
    {"1M_50pct_batch10K",  1000000, 10000, 0.50},
    {"1M_10pct_batch10K",  1000000, 10000, 0.10},
  };

  FILE* jf = fopen("bench_cagra_roaring_search.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"cagra_roaring_search\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n  \"n_sms\": %d,\n",
          prop.name, prop.multiProcessorCount);
  fprintf(jf, "  \"dim\": %d, \"k\": %d,\n", DIM, K);
  fprintf(jf, "  \"warmup\": %d, \"iters\": %d,\n", WARMUP, ITERS);
  fprintf(jf, "  \"results\": [\n");

  bool first_result = true;

  for (auto& cfg : configs) {
    int N = cfg.n_vectors;
    int NQ = cfg.n_queries;
    printf("=== %s (N=%d, NQ=%d, dim=%d, k=%d, pass=%.0f%%) ===\n",
           cfg.name, N, NQ, DIM, K, cfg.filter_pass_rate * 100);
    fflush(stdout);

    // Generate dataset
    auto dataset = raft::make_device_matrix<float, int64_t>(res, N, DIM);
    auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, DIM);
    {
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_data(static_cast<size_t>(N) * DIM);
      for (auto& v : h_data) v = fdist(rng);
      raft::update_device(dataset.data_handle(), h_data.data(), h_data.size(), stream);
      std::vector<float> h_queries(static_cast<size_t>(NQ) * DIM);
      for (auto& v : h_queries) v = fdist(rng);
      raft::update_device(queries.data_handle(), h_queries.data(), h_queries.size(), stream);
      raft::resource::sync_stream(res);
    }

    // Generate filter
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<uint32_t> pass_ids;
    for (int i = 0; i < N; ++i)
      if (dist(gen) < cfg.filter_pass_rate) pass_ids.push_back(static_cast<uint32_t>(i));
    printf("  Filter: %zu pass (%.2f%%)\n", pass_ids.size(), 100.0 * pass_ids.size() / N);

    // ---- Build filter variants ----

    // 1. Flat bitset
    raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
    {
      uint32_t n_words = (static_cast<uint32_t>(N) + 31) / 32;
      std::vector<uint32_t> h_bits(n_words, 0);
      for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
      raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
    }
    size_t bitset_bytes = (static_cast<size_t>(N) + 31) / 32 * sizeof(uint32_t);

    // 2. Roaring (default threshold — will have array containers at low density)
    auto gpu_roaring_default = cu_roaring::upload_from_sorted_ids(
      pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
      static_cast<uint32_t>(N));
    auto view_default = cu_roaring::make_view(gpu_roaring_default);

    // 3. Roaring promoted (all containers → bitmap, no array binary search)
    auto gpu_roaring_promoted = cu_roaring::upload_from_sorted_ids(
      pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
      static_cast<uint32_t>(N), 0, cu_roaring::PROMOTE_ALL);
    auto view_promoted = cu_roaring::make_view(gpu_roaring_promoted);

    // Memory stats
    auto roaring_bytes = [](const cu_roaring::GpuRoaring& g) -> size_t {
      return g.n_containers * (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint16_t))
           + static_cast<size_t>(g.n_bitmap_containers) * 1024 * sizeof(uint64_t);
      // Array/run data not tracked in struct fields — approximate
    };
    size_t meta_default  = roaring_bytes(gpu_roaring_default);
    size_t meta_promoted = roaring_bytes(gpu_roaring_promoted);

    printf("  Memory: bitset=%.2fKB  roaring=%.2fKB (%.1fx)  promoted=%.2fKB (%.1fx)\n",
           bitset_bytes / 1024.0,
           meta_default / 1024.0,  bitset_bytes / std::max(1.0, (double)meta_default),
           meta_promoted / 1024.0, bitset_bytes / std::max(1.0, (double)meta_promoted));
    printf("  Containers: default=%u (bmp=%u arr=%u)  promoted=%u (all bmp)\n",
           gpu_roaring_default.n_containers,
           gpu_roaring_default.n_bitmap_containers,
           gpu_roaring_default.n_array_containers,
           gpu_roaring_promoted.n_containers);

    // ---- Build CAGRA index ----
    printf("  Building CAGRA index...\n"); fflush(stdout);
    cuvs::neighbors::cagra::index_params build_params;
    build_params.graph_degree              = 32;
    build_params.intermediate_graph_degree = 48;
    auto cagra_index = cuvs::neighbors::cagra::build(
      res, build_params, raft::make_const_mdspan(dataset.view()));

    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, NQ, K);
    auto distances = raft::make_device_matrix<float, int64_t>(res, NQ, K);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = 256;

    // ---- Benchmark: No filter ----
    auto s_none = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), distances.view());
    });
    printf("  no_filter:        %.3f ms (std=%.3f)\n", s_none.median, s_none.std_dev);

    // ---- Benchmark: Bitset filter ----
    auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      flat_bitset.view());
    auto s_bitset = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), distances.view(), bitset_filt);
    });
    std::vector<uint32_t> r_bitset(static_cast<size_t>(NQ) * K);
    raft::update_host(r_bitset.data(), neighbors.data_handle(), r_bitset.size(), stream);
    raft::resource::sync_stream(res);
    printf("  bitset_filter:    %.3f ms (std=%.3f)\n", s_bitset.median, s_bitset.std_dev);

    // ---- Benchmark: Roaring warp (default containers) ----
    auto roaring_warp_filt = cuvs::neighbors::filtering::roaring_filter_warp(
      view_default, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));
    auto s_roaring_warp = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), distances.view(), roaring_warp_filt);
    });
    std::vector<uint32_t> r_roaring_warp(static_cast<size_t>(NQ) * K);
    raft::update_host(r_roaring_warp.data(), neighbors.data_handle(), r_roaring_warp.size(), stream);
    raft::resource::sync_stream(res);
    printf("  roaring_warp:     %.3f ms (std=%.3f)\n", s_roaring_warp.median, s_roaring_warp.std_dev);

    // ---- Benchmark: Roaring warp promoted (all bitmap) ----
    auto roaring_promoted_filt = cuvs::neighbors::filtering::roaring_filter_warp(
      view_promoted, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));
    auto s_roaring_promoted = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), distances.view(), roaring_promoted_filt);
    });
    std::vector<uint32_t> r_roaring_promoted(static_cast<size_t>(NQ) * K);
    raft::update_host(r_roaring_promoted.data(), neighbors.data_handle(), r_roaring_promoted.size(), stream);
    raft::resource::sync_stream(res);
    printf("  roaring_promoted: %.3f ms (std=%.3f)\n", s_roaring_promoted.median, s_roaring_promoted.std_dev);

    // ---- Recall ----
    double recall_bitset   = recall_at_k(r_bitset, r_bitset, K, NQ);
    double recall_warp     = recall_at_k(r_roaring_warp, r_bitset, K, NQ);
    double recall_promoted = recall_at_k(r_roaring_promoted, r_bitset, K, NQ);
    printf("  Recall@%d: bitset=%.4f  warp=%.4f  promoted=%.4f\n",
           K, recall_bitset, recall_warp, recall_promoted);

    // ---- Speedup summary ----
    double spd_warp     = s_bitset.median / s_roaring_warp.median;
    double spd_promoted = s_bitset.median / s_roaring_promoted.median;
    printf("  Speedup vs bitset: warp=%.2fx  promoted=%.2fx\n", spd_warp, spd_promoted);

    // ---- Throughput (QPS) ----
    double qps_none     = NQ / (s_none.median * 1e-3);
    double qps_bitset   = NQ / (s_bitset.median * 1e-3);
    double qps_warp     = NQ / (s_roaring_warp.median * 1e-3);
    double qps_promoted = NQ / (s_roaring_promoted.median * 1e-3);
    printf("  QPS: none=%.0f  bitset=%.0f  warp=%.0f  promoted=%.0f\n\n",
           qps_none, qps_bitset, qps_warp, qps_promoted);

    // ---- JSON ----
    if (!first_result) fprintf(jf, ",\n");
    first_result = false;
    fprintf(jf, "    {\n");
    fprintf(jf, "      \"config\": \"%s\",\n", cfg.name);
    fprintf(jf, "      \"n_vectors\": %d, \"n_queries\": %d,\n", N, NQ);
    fprintf(jf, "      \"filter_pass_rate\": %.4f, \"n_passing\": %zu,\n",
            cfg.filter_pass_rate, pass_ids.size());
    fprintf(jf, "      \"bitset_bytes\": %zu, \"roaring_default_bytes\": %zu, \"roaring_promoted_bytes\": %zu,\n",
            bitset_bytes, meta_default, meta_promoted);
    fprintf(jf, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u,\n",
            gpu_roaring_default.n_containers,
            gpu_roaring_default.n_bitmap_containers,
            gpu_roaring_default.n_array_containers);
    write_stats_json(jf, "no_filter", s_none);
    write_stats_json(jf, "bitset", s_bitset);
    write_stats_json(jf, "roaring_warp", s_roaring_warp);
    write_stats_json(jf, "roaring_promoted", s_roaring_promoted);
    fprintf(jf, "      \"recall_bitset\": %.4f,\n", recall_bitset);
    fprintf(jf, "      \"recall_roaring_warp\": %.4f,\n", recall_warp);
    fprintf(jf, "      \"recall_roaring_promoted\": %.4f,\n", recall_promoted);
    fprintf(jf, "      \"speedup_warp_vs_bitset\": %.4f,\n", spd_warp);
    fprintf(jf, "      \"speedup_promoted_vs_bitset\": %.4f,\n", spd_promoted);
    fprintf(jf, "      \"qps_none\": %.0f, \"qps_bitset\": %.0f, \"qps_warp\": %.0f, \"qps_promoted\": %.0f\n",
            qps_none, qps_bitset, qps_warp, qps_promoted);
    fprintf(jf, "    }");

    cu_roaring::gpu_roaring_free(gpu_roaring_default);
    cu_roaring::gpu_roaring_free(gpu_roaring_promoted);
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  printf("=== COMPLETE — results at bench_cagra_roaring_search.json ===\n");
  return 0;
}
