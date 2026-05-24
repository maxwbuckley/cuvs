/*
 * Throughput benchmark: CAGRA filtered search QPS at various batch sizes.
 * Measures saturated GPU throughput for comparison with VecFlow (5M QPS @ 90% recall).
 */
#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/utils.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
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

int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB)\n\n", prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0));

  constexpr int N       = 1000000;
  constexpr int DIM     = 128;
  constexpr int K       = 10;
  constexpr int WARMUP  = 10;
  constexpr int ITERS   = 30;

  int batch_sizes[] = {1, 10, 100, 1000, 10000};
  double pass_rates[] = {0.50, 0.10, 0.01};

  // Generate dataset
  printf("Generating dataset: N=%d, dim=%d\n", N, DIM);
  fflush(stdout);
  auto dataset = raft::make_device_matrix<float, int64_t>(res, N, DIM);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::vector<float> h_data(static_cast<size_t>(N) * DIM);
    for (auto& v : h_data) v = fdist(rng);
    raft::update_device(dataset.data_handle(), h_data.data(), h_data.size(), stream);
    raft::resource::sync_stream(res);
  }

  // Build CAGRA index (once)
  printf("Building CAGRA index...\n");
  fflush(stdout);
  cuvs::neighbors::cagra::index_params build_params;
  build_params.graph_degree              = 32;
  build_params.intermediate_graph_degree = 48;
  auto cagra_index = cuvs::neighbors::cagra::build(
    res, build_params, raft::make_const_mdspan(dataset.view()));
  printf("Index built.\n\n");
  fflush(stdout);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = 64;

  // Open JSON
  FILE* jf = fopen("bench_throughput.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"cagra_throughput\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
  fprintf(jf, "  \"n_vectors\": %d, \"dim\": %d, \"k\": %d,\n", N, DIM, K);
  fprintf(jf, "  \"warmup\": %d, \"iters\": %d,\n", WARMUP, ITERS);
  fprintf(jf, "  \"results\": [\n");
  bool first = true;

  printf("%-12s %-10s %-12s %-12s %-12s %-12s %-12s\n",
         "batch_size", "pass_rate", "no_filt_qps", "bitset_qps", "roaring_qps", "warp_qps", "warp_ms");
  printf("%-12s %-10s %-12s %-12s %-12s %-12s %-12s\n",
         "----------", "---------", "-----------", "----------", "-----------", "--------", "-------");
  fflush(stdout);

  for (int nq : batch_sizes) {
    // Generate queries for this batch size
    auto queries = raft::make_device_matrix<float, int64_t>(res, nq, DIM);
    {
      std::mt19937 rng(99 + nq);
      std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
      std::vector<float> hq(static_cast<size_t>(nq) * DIM);
      for (auto& v : hq) v = fdist(rng);
      raft::update_device(queries.data_handle(), hq.data(), hq.size(), stream);
      raft::resource::sync_stream(res);
    }

    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, nq, K);
    auto distances = raft::make_device_matrix<float, int64_t>(res, nq, K);

    // No filter baseline
    auto s_none = bench_gpu(WARMUP, ITERS, [&]() {
      cuvs::neighbors::cagra::search(
        res, search_params, cagra_index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), distances.view());
    });
    double qps_none = nq / (s_none.median / 1000.0);

    for (double pass_rate : pass_rates) {
      // Generate filter
      std::mt19937 gen(123);
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      std::vector<uint32_t> pass_ids;
      for (int i = 0; i < N; ++i) {
        if (dist(gen) < pass_rate) pass_ids.push_back(i);
      }

      // Roaring
      auto gpu_roaring = cu_roaring::upload_from_sorted_ids(
        pass_ids.data(), static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));
      cu_roaring::build_key_bloom(gpu_roaring, stream);
      auto roaring_view = cu_roaring::make_view(gpu_roaring);

      // Flat bitset
      raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
      {
        uint32_t n_words = (N + 31) / 32;
        std::vector<uint32_t> h_bits(n_words, 0);
        for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
        raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
      }

      auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
        flat_bitset.view());
      auto roaring_filt = cuvs::neighbors::filtering::roaring_filter(
        roaring_view, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));
      auto warp_filt = cuvs::neighbors::filtering::roaring_filter_warp(
        roaring_view, static_cast<uint32_t>(pass_ids.size()), static_cast<uint32_t>(N));

      // Bitset
      auto s_bitset = bench_gpu(WARMUP, ITERS, [&]() {
        cuvs::neighbors::cagra::search(
          res, search_params, cagra_index,
          raft::make_const_mdspan(queries.view()),
          neighbors.view(), distances.view(), bitset_filt);
      });

      // Roaring
      auto s_roaring = bench_gpu(WARMUP, ITERS, [&]() {
        cuvs::neighbors::cagra::search(
          res, search_params, cagra_index,
          raft::make_const_mdspan(queries.view()),
          neighbors.view(), distances.view(), roaring_filt);
      });

      // Warp
      auto s_warp = bench_gpu(WARMUP, ITERS, [&]() {
        cuvs::neighbors::cagra::search(
          res, search_params, cagra_index,
          raft::make_const_mdspan(queries.view()),
          neighbors.view(), distances.view(), warp_filt);
      });

      double qps_bitset  = nq / (s_bitset.median / 1000.0);
      double qps_roaring = nq / (s_roaring.median / 1000.0);
      double qps_warp    = nq / (s_warp.median / 1000.0);

      printf("%-12d %-10.0f%% %-12.0f %-12.0f %-12.0f %-12.0f %-12.3f\n",
             nq, pass_rate * 100, qps_none, qps_bitset, qps_roaring, qps_warp, s_warp.median);
      fflush(stdout);

      if (!first) fprintf(jf, ",\n");
      first = false;
      fprintf(jf, "    {\"batch_size\": %d, \"pass_rate\": %.2f, ", nq, pass_rate);
      fprintf(jf, "\"no_filter_qps\": %.0f, ", qps_none);
      fprintf(jf, "\"bitset_qps\": %.0f, ", qps_bitset);
      fprintf(jf, "\"roaring_qps\": %.0f, ", qps_roaring);
      fprintf(jf, "\"roaring_warp_qps\": %.0f, ", qps_warp);
      fprintf(jf, "\"no_filter_ms\": %.4f, ", s_none.median);
      fprintf(jf, "\"bitset_ms\": %.4f, ", s_bitset.median);
      fprintf(jf, "\"roaring_ms\": %.4f, ", s_roaring.median);
      fprintf(jf, "\"roaring_warp_ms\": %.4f}", s_warp.median);

      cu_roaring::gpu_roaring_free(gpu_roaring);
    }
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);

  printf("\n=== THROUGHPUT BENCHMARK COMPLETE ===\n");
  return 0;
}
