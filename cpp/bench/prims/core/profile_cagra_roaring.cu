/*
 * Minimal CAGRA profiling harness for nsight compute.
 *
 * Runs ONE bitset search + ONE roaring search after warmup,
 * with cudaProfilerStart/Stop bracketing so ncu captures only the
 * measured launches.
 *
 * Build:
 *   cd cpp/bench/prims/core/build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release && make -j profile_cagra_roaring
 *
 * Profile:
 *   ncu --profile-from-start off --set full \
 *       --target-processes all \
 *       -o profile_cagra_roaring \
 *       ./profile_cagra_roaring
 */

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/make_view.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/bitset.cuh>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

int main(int argc, char** argv)
{
  // Config: 1M at 50% pass rate (largest speedup)
  constexpr int N   = 1000000;
  constexpr int DIM = 128;
  constexpr int NQ  = 100;
  constexpr int K   = 10;
  constexpr double PASS_RATE = 0.50;
  constexpr int WARMUP = 10;

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("Config: N=%d dim=%d NQ=%d K=%d pass=%.0f%%\n",
         N, DIM, NQ, K, PASS_RATE * 100);

  // --- Dataset ---
  printf("Generating dataset...\n");
  auto dataset = raft::make_device_matrix<float, int64_t>(res, N, DIM);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::vector<float> h_data(static_cast<size_t>(N) * DIM);
    for (auto& v : h_data) v = fdist(rng);
    raft::update_device(dataset.data_handle(), h_data.data(), h_data.size(), stream);
    raft::resource::sync_stream(res);
  }

  // --- CAGRA index ---
  printf("Building CAGRA index...\n");
  cuvs::neighbors::cagra::index_params build_params;
  build_params.graph_degree              = 32;
  build_params.intermediate_graph_degree = 48;
  auto index = cuvs::neighbors::cagra::build(
    res, build_params, raft::make_const_mdspan(dataset.view()));

  // --- Queries ---
  auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, DIM);
  {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::vector<float> h_q(static_cast<size_t>(NQ) * DIM);
    for (auto& v : h_q) v = fdist(rng);
    raft::update_device(queries.data_handle(), h_q.data(), h_q.size(), stream);
    raft::resource::sync_stream(res);
  }

  // --- Filter IDs ---
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<uint32_t> pass_ids;
  for (int i = 0; i < N; ++i)
    if (dist(gen) < PASS_RATE) pass_ids.push_back(static_cast<uint32_t>(i));
  printf("Filter: %zu pass (%.2f%%)\n", pass_ids.size(),
         100.0 * pass_ids.size() / N);

  // --- Bitset filter ---
  raft::core::bitset<uint32_t, int64_t> flat_bitset(res, static_cast<int64_t>(N), false);
  {
    uint32_t n_words = (static_cast<uint32_t>(N) + 31) / 32;
    std::vector<uint32_t> h_bits(n_words, 0);
    for (auto id : pass_ids) h_bits[id / 32] |= (1u << (id % 32));
    raft::update_device(flat_bitset.data(), h_bits.data(), n_words, stream);
  }
  auto bitset_filt = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(
    flat_bitset.view());

  // --- Roaring filter ---
  auto gpu_roaring = cu_roaring::upload_from_sorted_ids(
    pass_ids.data(), static_cast<uint32_t>(pass_ids.size()),
    static_cast<uint32_t>(N));
  auto roaring_filt = cuvs::neighbors::filtering::roaring_filter(gpu_roaring);

  // --- Output buffers ---
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, NQ, K);
  auto distances = raft::make_device_matrix<float, int64_t>(res, NQ, K);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = 256;

  auto q_view = raft::make_const_mdspan(queries.view());

  // --- Warmup (NOT profiled) ---
  printf("Warming up (%d iterations each)...\n", WARMUP);
  for (int i = 0; i < WARMUP; ++i) {
    cuvs::neighbors::cagra::search(res, search_params, index, q_view,
                                    neighbors.view(), distances.view(), bitset_filt);
    cuvs::neighbors::cagra::search(res, search_params, index, q_view,
                                    neighbors.view(), distances.view(), roaring_filt);
  }
  cudaDeviceSynchronize();

  // --- Profiled region ---
  printf("Starting profiled region...\n");
  cudaProfilerStart();

  // Bitset search
  cuvs::neighbors::cagra::search(res, search_params, index, q_view,
                                  neighbors.view(), distances.view(), bitset_filt);
  cudaDeviceSynchronize();

  // Roaring search
  cuvs::neighbors::cagra::search(res, search_params, index, q_view,
                                  neighbors.view(), distances.view(), roaring_filt);
  cudaDeviceSynchronize();

  cudaProfilerStop();
  printf("Profiling complete.\n");

  cu_roaring::gpu_roaring_free(gpu_roaring);
  return 0;
}
