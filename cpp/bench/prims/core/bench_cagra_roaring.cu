/*
 * Benchmark: CAGRA + Brute Force search with roaring_filter vs bitset_filter.
 *
 * Measures search latency and recall at various selectivities,
 * comparing native Roaring filter (no decompression) vs flat bitset filter.
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

// Only include what we need — avoid cu_roaring.cuh which pulls in CRoaring headers
#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/utils.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
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

// Forward declarations from cu_roaring library (avoids CRoaring header dependency)
namespace cu_roaring {
void build_key_bloom(GpuRoaring& bitmap, cudaStream_t stream);
void gpu_roaring_free(GpuRoaring& bitmap);
}

// ============================================================================
// Helpers
// ============================================================================
struct Stats {
  double median, mean, min_v, max_v, std_dev;
};

static Stats compute_stats(std::vector<double>& t)
{
  std::sort(t.begin(), t.end());
  int n      = t.size();
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, t[0], t[n - 1], std::sqrt(var / n)};
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

// Compute recall@k
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

  // ========================================================
  // Parameters
  // ========================================================
  constexpr int DIM       = 128;
  constexpr int K         = 10;
  constexpr int N_QUERIES = 100;
  constexpr int WARMUP    = 10;   // absorb JIT compilation
  constexpr int ITERS     = 30;   // n>=30 per benchmarking rules

  struct Config {
    const char* name;
    int n_vectors;
    double filter_pass_rate;  // fraction of vectors that PASS the filter
  };

  Config configs[] = {
    // Various selectivities (pass_rate = 1 - selectivity)
    {"1M, 50% pass",    1000000,  0.50},
    {"1M, 10% pass",    1000000,  0.10},
    {"1M, 1% pass",     1000000,  0.01},
  };

  // Open results file
  FILE* jf = fopen("results/raw/bench_cagra_roaring_search.json", "w");
  if (!jf) jf = fopen("bench_cagra_roaring_search.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"cagra_roaring_search\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
  fprintf(jf, "  \"dim\": %d, \"k\": %d, \"n_queries\": %d,\n", DIM, K, N_QUERIES);
  fprintf(jf, "  \"results\": [\n");

  bool first_result = true;

  for (auto& cfg : configs) {
    int N = cfg.n_vectors;
    printf("=== %s (N=%d, dim=%d, k=%d) ===\n", cfg.name, N, DIM, K);
    fflush(stdout);

    // Generate random dataset
    printf("  Generating dataset...\n");
    fflush(stdout);
    auto dataset = raft::make_device_matrix<float, int64_t>(res, N, DIM);
    auto queries = raft::make_device_matrix<float, int64_t>(res, N_QUERIES, DIM);

    // Fill with random data using thrust
    {
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> h_data(static_cast<size_t>(N) * DIM);
      for (auto& v : h_data) v = fdist(rng);
      raft::update_device(dataset.data_handle(), h_data.data(), h_data.size(), stream);

      std::vector<float> h_queries(static_cast<size_t>(N_QUERIES) * DIM);
      for (auto& v : h_queries) v = fdist(rng);
      raft::update_device(queries.data_handle(), h_queries.data(), h_queries.size(), stream);
      raft::resource::sync_stream(res);
    }

    // Generate filter bitmap (on host, then upload)
    printf("  Generating filter (%.1f%% pass rate)...\n", cfg.filter_pass_rate * 100);
    fflush(stdout);
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<uint32_t> pass_ids;
    for (int i = 0; i < N; ++i) {
      if (dist(gen) < cfg.filter_pass_rate) pass_ids.push_back(i);
    }
    printf("  Filter: %zu vectors pass (%.2f%%)\n",
           pass_ids.size(), 100.0 * pass_ids.size() / N);

    // Create GPU Roaring bitmap
    auto gpu_roaring = cu_roaring::upload_from_sorted_ids(
      pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
      static_cast<uint32_t>(N));
    cu_roaring::build_key_bloom(gpu_roaring, stream);
    auto roaring_view = cu_roaring::make_view(gpu_roaring);

    // Create flat bitset
    raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
    // Set bits for pass_ids
    {
      uint32_t n_words = (N + 31) / 32;
      std::vector<uint32_t> h_bits(n_words, 0);
      for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
      raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
    }

    printf("  Roaring: %.2f MB (%u containers)  Flat: %.2f MB  Ratio: %.1fx\n",
           gpu_roaring.n_containers * 8.0 / 1024 / 1024,  // approximate
           gpu_roaring.n_containers,
           (N / 8.0) / 1024 / 1024,
           (N / 8.0) / std::max(1.0, gpu_roaring.n_containers * 8.0));

    // ====================================
    // Build CAGRA index
    // ====================================
    printf("  Building CAGRA index...\n");
    fflush(stdout);

    cuvs::neighbors::cagra::index_params build_params;
    build_params.graph_degree          = 32;
    build_params.intermediate_graph_degree = 48;

    auto cagra_index = cuvs::neighbors::cagra::build(
      res,
      build_params,
      raft::make_const_mdspan(dataset.view()));

    printf("  CAGRA index built: graph_degree=%d\n", build_params.graph_degree);
    fflush(stdout);

    // Allocate output
    auto neighbors_cagra = raft::make_device_matrix<uint32_t, int64_t>(res, N_QUERIES, K);
    auto distances_cagra = raft::make_device_matrix<float, int64_t>(res, N_QUERIES, K);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = 256;

    // ====================================
    // CAGRA + No filter (baseline)
    // ====================================
    printf("  Benchmarking CAGRA...\n");
    fflush(stdout);

    auto s_none = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_cagra.view(), distances_cagra.view());
    });
    printf("    No filter:      median=%.3f mean=%.3f std=%.3f ms\n",
           s_none.median, s_none.mean, s_none.std_dev);

    // Get ground truth (no filter) for recall computation
    std::vector<uint32_t> gt_none(N_QUERIES * K);
    raft::update_host(gt_none.data(), neighbors_cagra.data_handle(), N_QUERIES * K, stream);
    raft::resource::sync_stream(res);

    // ====================================
    // CAGRA + bitset_filter (flat bitset)
    // ====================================
    auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
      flat_bitset.view());

    auto s_bitset = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_cagra.view(), distances_cagra.view(),
        bitset_filt);
    });

    std::vector<uint32_t> r_bitset(N_QUERIES * K);
    raft::update_host(r_bitset.data(), neighbors_cagra.data_handle(), N_QUERIES * K, stream);
    raft::resource::sync_stream(res);
    printf("    bitset_filter:  median=%.3f mean=%.3f std=%.3f ms\n",
           s_bitset.median, s_bitset.mean, s_bitset.std_dev);

    // ====================================
    // CAGRA + roaring_filter (native, no decompress)
    // ====================================
    auto roaring_filt = cuvs::neighbors::filtering::roaring_filter(
      roaring_view, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));

    auto s_roaring = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_cagra.view(), distances_cagra.view(),
        roaring_filt);
    });

    std::vector<uint32_t> r_roaring(N_QUERIES * K);
    raft::update_host(r_roaring.data(), neighbors_cagra.data_handle(), N_QUERIES * K, stream);
    raft::resource::sync_stream(res);
    printf("    roaring_filter: median=%.3f mean=%.3f std=%.3f ms\n",
           s_roaring.median, s_roaring.mean, s_roaring.std_dev);

    // ====================================
    // CAGRA + roaring_filter_warp
    // ====================================
    auto roaring_warp_filt = cuvs::neighbors::filtering::roaring_filter_warp(
      roaring_view, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));

    auto s_roaring_warp = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_cagra.view(), distances_cagra.view(),
        roaring_warp_filt);
    });

    std::vector<uint32_t> r_roaring_warp(N_QUERIES * K);
    raft::update_host(r_roaring_warp.data(), neighbors_cagra.data_handle(), N_QUERIES * K, stream);
    raft::resource::sync_stream(res);
    printf("    roaring_warp:   median=%.3f mean=%.3f std=%.3f ms\n",
           s_roaring_warp.median, s_roaring_warp.mean, s_roaring_warp.std_dev);

    // ====================================
    // Compute recall (all filtered results vs bitset_filter as ground truth)
    // ====================================
    double recall_bitset  = recall_at_k(r_bitset, r_bitset, K, N_QUERIES);
    double recall_roaring = recall_at_k(r_roaring, r_bitset, K, N_QUERIES);
    double recall_warp    = recall_at_k(r_roaring_warp, r_bitset, K, N_QUERIES);

    printf("    Recall@%d (vs bitset_filter ground truth):\n", K);
    printf("      bitset_filter:  %.4f\n", recall_bitset);
    printf("      roaring_filter: %.4f\n", recall_roaring);
    printf("      roaring_warp:   %.4f\n", recall_warp);

    // ====================================
    // Brute Force search
    // ====================================
    printf("  Benchmarking Brute Force...\n");
    fflush(stdout);

    auto bf_index = cuvs::neighbors::brute_force::build(
      res, raft::make_const_mdspan(dataset.view()),
      cuvs::distance::DistanceType::L2Expanded);

    auto neighbors_bf = raft::make_device_matrix<int64_t, int64_t>(res, N_QUERIES, K);
    auto distances_bf = raft::make_device_matrix<float, int64_t>(res, N_QUERIES, K);

    cuvs::neighbors::brute_force::search_params bf_params;

    auto s_bf_none = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::brute_force::search(
        res, bf_params, bf_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_bf.view(), distances_bf.view());
    });
    printf("    BF no filter:      median=%.3f mean=%.3f std=%.3f ms\n",
           s_bf_none.median, s_bf_none.mean, s_bf_none.std_dev);

    auto s_bf_bitset = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::brute_force::search(
        res, bf_params, bf_index,
        raft::make_const_mdspan(queries.view()),
        neighbors_bf.view(), distances_bf.view(),
        bitset_filt);
    });
    printf("    BF bitset_filter:  median=%.3f mean=%.3f std=%.3f ms\n",
           s_bf_bitset.median, s_bf_bitset.mean, s_bf_bitset.std_dev);

    // Note: brute_force may not have roaring_filter template instantiations
    // If it crashes, we skip it
    printf("\n");

    // Write JSON
    if (!first_result) fprintf(jf, ",\n");
    first_result = false;
    fprintf(jf, "    {\n");
    fprintf(jf, "      \"config\": \"%s\",\n", cfg.name);
    fprintf(jf, "      \"n_vectors\": %d,\n", N);
    fprintf(jf, "      \"filter_pass_rate\": %.4f,\n", cfg.filter_pass_rate);
    fprintf(jf, "      \"n_passing\": %zu,\n", pass_ids.size());
    fprintf(jf, "      \"warmup\": %d, \"iters\": %d,\n", WARMUP, ITERS);
    fprintf(jf, "      \"cagra_no_filter_median_ms\": %.4f,\n", s_none.median);
    fprintf(jf, "      \"cagra_no_filter_mean_ms\": %.4f,\n", s_none.mean);
    fprintf(jf, "      \"cagra_no_filter_std_ms\": %.4f,\n", s_none.std_dev);
    fprintf(jf, "      \"cagra_bitset_median_ms\": %.4f,\n", s_bitset.median);
    fprintf(jf, "      \"cagra_bitset_mean_ms\": %.4f,\n", s_bitset.mean);
    fprintf(jf, "      \"cagra_bitset_std_ms\": %.4f,\n", s_bitset.std_dev);
    fprintf(jf, "      \"cagra_roaring_median_ms\": %.4f,\n", s_roaring.median);
    fprintf(jf, "      \"cagra_roaring_mean_ms\": %.4f,\n", s_roaring.mean);
    fprintf(jf, "      \"cagra_roaring_std_ms\": %.4f,\n", s_roaring.std_dev);
    fprintf(jf, "      \"cagra_roaring_warp_median_ms\": %.4f,\n", s_roaring_warp.median);
    fprintf(jf, "      \"cagra_roaring_warp_mean_ms\": %.4f,\n", s_roaring_warp.mean);
    fprintf(jf, "      \"cagra_roaring_warp_std_ms\": %.4f,\n", s_roaring_warp.std_dev);
    fprintf(jf, "      \"recall_bitset\": %.4f,\n", recall_bitset);
    fprintf(jf, "      \"recall_roaring\": %.4f,\n", recall_roaring);
    fprintf(jf, "      \"recall_roaring_warp\": %.4f,\n", recall_warp);
    fprintf(jf, "      \"bf_no_filter_median_ms\": %.4f,\n", s_bf_none.median);
    fprintf(jf, "      \"bf_bitset_median_ms\": %.4f\n", s_bf_bitset.median);
    fprintf(jf, "    }");

    cu_roaring::gpu_roaring_free(gpu_roaring);
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);

  printf("\n=== ALL BENCHMARKS COMPLETE ===\n");
  return 0;
}
