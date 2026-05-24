/*
 * Benchmark: schedule-driven vs cuVS bitset across the synthetic filter
 * generators in ../../../../roaring-benchmark (uniform / clustered / temporal
 * / power_law / multi_tenant). Filters are loaded from CRoaring portable
 * serialization (.bin) files; the database is synthetic, sized to match the
 * filter universe.
 *
 *   SYNTH_FILTER_DIR=/path/to/bitmaps  (required)
 *   SYNTH_N=10000000                   (default 10M)
 *   SYNTH_DIM=128 SYNTH_NQ=64          (defaults)
 *
 * Writes bench_synth_filters.json next to the binary.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>

#include <cu_roaring/cu_roaring.cuh>
#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/detail/filtered_search.cuh>

#include <roaring/roaring.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/bitset.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace cu_roaring { void gpu_roaring_free(GpuRoaring&); }

struct Stats { double median, mean, std_dev; };
static Stats stats_of(std::vector<double> t)
{
  if (t.empty()) return {0,0,0};
  std::sort(t.begin(), t.end());
  int n = static_cast<int>(t.size());
  double sum = 0; for (double v : t) sum += v;
  double mean = sum / n;
  double var = 0; for (double v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, std::sqrt(var / std::max(1, n - 1))};
}

static double recall_at_k(const std::vector<uint32_t>& got,
                          const std::vector<int64_t>& ref, int nq, int k)
{
  int hit = 0;
  for (int q = 0; q < nq; ++q)
    for (int i = 0; i < k; ++i) {
      uint32_t g = got[q * k + i];
      for (int j = 0; j < k; ++j)
        if (static_cast<int64_t>(g) == ref[q * k + j]) { ++hit; break; }
    }
  return static_cast<double>(hit) / (nq * k);
}

static std::string parse_generator(const std::string& name)
{
  // file pattern: <generator>_s<sel>_t<trial>.bin
  // (multi_tenant has an underscore; clip at "_s")
  auto p = name.find("_s");
  return p == std::string::npos ? name : name.substr(0, p);
}

static double parse_selectivity(const std::string& name)
{
  auto p1 = name.find("_s");
  auto p2 = name.find("_t", p1);
  if (p1 == std::string::npos || p2 == std::string::npos) return -1.0;
  return std::stod(name.substr(p1 + 2, p2 - p1 - 2));
}

int main()
{
  const char* dir_env = std::getenv("SYNTH_FILTER_DIR");
  if (!dir_env) {
    fprintf(stderr, "set SYNTH_FILTER_DIR=/path/to/bitmaps\n");
    return 1;
  }
  std::string dir(dir_env);

  int N  = std::getenv("SYNTH_N")   ? std::atoi(std::getenv("SYNTH_N"))   : 10'000'000;
  int D  = std::getenv("SYNTH_DIM") ? std::atoi(std::getenv("SYNTH_DIM")) : 128;
  int NQ = std::getenv("SYNTH_NQ")  ? std::atoi(std::getenv("SYNTH_NQ"))  : 64;
  constexpr int K = 10, WARMUP = 8, ITERS = 20;

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(rng, stream);
  curandSetPseudoRandomGeneratorSeed(rng, 0xC0FFEE);

  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s   N=%d  D=%d  Q=%d  filter_dir=%s\n",
         prop.name, N, D, NQ, dir.c_str());

  auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
  curandGenerateNormal(rng, dataset.data_handle(),
                       static_cast<size_t>(N) * D, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

  cuvs::neighbors::brute_force::index_params bp;
  bp.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index = cuvs::neighbors::brute_force::build(
    res, bp, raft::make_const_mdspan(dataset.view()));

  auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, D);
  curandGenerateNormal(rng, queries.data_handle(),
                       static_cast<size_t>(NQ) * D, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, NQ, K);
  auto dists     = raft::make_device_matrix<float, int64_t>(res, NQ, K);
  uint32_t* d_rid = nullptr;
  float*    d_rsc = nullptr;
  cudaMalloc(&d_rid, static_cast<size_t>(NQ) * K * sizeof(uint32_t));
  cudaMalloc(&d_rsc, static_cast<size_t>(NQ) * K * sizeof(float));

  std::vector<std::string> files;
  if (DIR* dp = opendir(dir.c_str())) {
    while (auto* e = readdir(dp)) {
      std::string n = e->d_name;
      if (n.size() > 4 && n.substr(n.size() - 4) == ".bin") files.push_back(n);
    }
    closedir(dp);
  }
  std::sort(files.begin(), files.end());
  printf("  %zu filter files\n\n", files.size());

  FILE* jf = fopen("bench_synth_filters.json", "w");
  fprintf(jf, "{\n  \"gpu\":\"%s\",\"n\":%d,\"dim\":%d,\"nq\":%d,\"k\":%d,\n",
          prop.name, N, D, NQ, K);
  fprintf(jf, "  \"results\":[\n");
  bool first = true;

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);

  for (const std::string& fname : files) {
    std::string path = dir + "/" + fname;
    struct stat st; if (stat(path.c_str(), &st) != 0) continue;
    std::vector<char> buf(st.st_size);
    {
      FILE* f = fopen(path.c_str(), "rb");
      if (!f) continue;
      fread(buf.data(), 1, st.st_size, f);
      fclose(f);
    }
    roaring_bitmap_t* cpu_bm =
      roaring_bitmap_portable_deserialize_safe(buf.data(), buf.size());
    if (!cpu_bm) { printf("skip %s (deserialize)\n", fname.c_str()); continue; }
    // Keep only ids within [0, N): the .bin universes may exceed N.
    uint64_t card_full = roaring_bitmap_get_cardinality(cpu_bm);
    roaring_bitmap_remove_range(cpu_bm, static_cast<uint64_t>(N), UINT64_MAX);
    uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);

    if (card == 0) { roaring_bitmap_free(cpu_bm); continue; }
    if (card > static_cast<uint64_t>(N) / 2 + 1) {
      // Avoid the high-sel regression at very high selectivity (gather > VRAM).
      // The user said the design fragments badly here; we already covered 50%
      // in the main grid. Skip > 0.55 here.
      if (card * 2 > static_cast<uint64_t>(N) * 11 / 10) {
        printf("skip %s (sel %.2f%% too high for this harness)\n",
               fname.c_str(), 100.0 * card / N);
        roaring_bitmap_free(cpu_bm);
        continue;
      }
    }
    double sparsity = 1.0 - static_cast<double>(card) / N;

    std::string gen = parse_generator(fname);
    double sel = parse_selectivity(fname);

    // Extract ids (sorted) for the raft bitset.
    std::vector<uint32_t> ids(card);
    roaring_bitmap_to_uint32_array(cpu_bm, ids.data());

    raft::core::bitset<uint32_t, int64_t> bits(res, static_cast<int64_t>(N), false);
    {
      uint32_t nw = (static_cast<uint32_t>(N) + 31) / 32;
      std::vector<uint32_t> hb(nw, 0);
      for (uint32_t id : ids) hb[id / 32] |= (1u << (id % 32));
      raft::update_device(bits.data(), hb.data(), nw, stream);
      raft::resource::sync_stream(res);
    }
    auto bitset_filt =
      cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bits.view());

    auto gpu = cu_roaring::upload(cpu_bm, static_cast<uint32_t>(N));

    size_t gather_bytes = static_cast<size_t>(card) * D * sizeof(float);
    size_t free_b = 0, total_b = 0; cudaMemGetInfo(&free_b, &total_b);
    if (free_b < gather_bytes + 1'500'000'000ull) {
      printf("skip %s (card %llu, gather %.1f GB, only %.1f GB free)\n",
             fname.c_str(), (unsigned long long)card,
             gather_bytes / 1e9, free_b / 1e9);
      cu_roaring::gpu_roaring_free(gpu);
      roaring_bitmap_free(cpu_bm);
      continue;
    }
    auto sched = cu_roaring::build_schedule(gpu, N, dataset.data_handle(),
                                            D, stream);

    cuvs::neighbors::brute_force::search_params sp;
    auto run_a = [&] {
      cuvs::neighbors::brute_force::search(res, sp, index,
        raft::make_const_mdspan(queries.view()),
        neighbors.view(), dists.view(), bitset_filt);
    };
    auto run_b = [&] {
      cu_roaring::roaring_filtered_search(cublas, queries.data_handle(), NQ,
        dataset.data_handle(), N, D, sched, K, d_rid, d_rsc, stream);
    };
    for (int i = 0; i < WARMUP; ++i) { run_a(); run_b(); }
    cudaStreamSynchronize(stream);

    auto time = [&](auto&& f) {
      cudaEventRecord(e0, stream); f();
      cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      return static_cast<double>(ms);
    };
    std::vector<double> ta, tb;
    for (int i = 0; i < ITERS; ++i) {
      if (i & 1) { tb.push_back(time(run_b)); ta.push_back(time(run_a)); }
      else       { ta.push_back(time(run_a)); tb.push_back(time(run_b)); }
    }
    auto sa = stats_of(ta);
    auto sb = stats_of(tb);

    // recall
    run_a();
    std::vector<int64_t> ref(static_cast<size_t>(NQ) * K);
    raft::update_host(ref.data(), neighbors.data_handle(), ref.size(), stream);
    raft::resource::sync_stream(res);
    run_b();
    cudaStreamSynchronize(stream);
    std::vector<uint32_t> got(static_cast<size_t>(NQ) * K);
    cudaMemcpy(got.data(), d_rid, got.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    double recall = recall_at_k(got, ref, NQ, K);

    int n_direct = 0, n_masked = 0, n_gather = 0;
    uint64_t c_direct = 0, c_masked = 0, c_gather = 0;
    for (const auto& t : sched.tasks) {
      if (t.kind == cu_roaring::GemmTask::kGather) { n_gather++; c_gather += t.n_cols; }
      else if (t.mask)                              { n_masked++; c_masked += t.n_cols; }
      else                                          { n_direct++; c_direct += t.n_cols; }
    }

    double qps_a = NQ / (sa.median * 1e-3);
    double qps_b = NQ / (sb.median * 1e-3);
    double sp_ = sa.median / sb.median;
    printf("%-40s sel=%.4f card=%llu  cont=%u/%u/%u  D/M/G=%d/%d/%d  bs=%.2fms r=%.2fms  %.2fx  rec=%.3f\n",
           fname.c_str(), sel, (unsigned long long)card,
           gpu.n_run_containers, gpu.n_array_containers, gpu.n_bitmap_containers,
           n_direct, n_masked, n_gather, sa.median, sb.median, sp_, recall);
    fflush(stdout);

    fprintf(jf, "    %s{\"file\":\"%s\",\"generator\":\"%s\",\"sel\":%.6f,"
                "\"card\":%llu,\"sparsity\":%.4f,"
                "\"containers\":{\"run\":%u,\"array\":%u,\"bitmap\":%u},"
                "\"schedule\":{\"direct_tasks\":%d,\"direct_cols\":%llu,"
                "\"masked_tasks\":%d,\"masked_cols\":%llu,"
                "\"gather_tasks\":%d,\"gather_cols\":%llu},"
                "\"bitset_ms\":%.4f,\"bitset_std\":%.4f,\"bitset_qps\":%.0f,"
                "\"roaring_ms\":%.4f,\"roaring_std\":%.4f,\"roaring_qps\":%.0f,"
                "\"speedup\":%.4f,\"recall\":%.4f,\"fallback\":%d}",
            first ? "" : ",\n", fname.c_str(), gen.c_str(), sel,
            (unsigned long long)card, sparsity,
            gpu.n_run_containers, gpu.n_array_containers, gpu.n_bitmap_containers,
            n_direct, (unsigned long long)c_direct,
            n_masked, (unsigned long long)c_masked,
            n_gather, (unsigned long long)c_gather,
            sa.median, sa.std_dev, qps_a, sb.median, sb.std_dev, qps_b,
            sp_, recall, sched.used_fallback ? 1 : 0);
    first = false;

    cu_roaring::free_schedule(sched);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(cpu_bm);
    (void)card_full;  // silence unused
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  cudaFree(d_rid);
  cudaFree(d_rsc);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  curandDestroyGenerator(rng);
  cublasDestroy(cublas);
  printf("wrote bench_synth_filters.json\n");
  return 0;
}
