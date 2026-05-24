/*
 * Benchmark: schedule-driven roaring-filtered search vs cuVS bitset filter.
 *
 *   A = cuVS brute_force::search + raft bitset_filter (production)
 *   B = cu_roaring::roaring_filtered_search (schedule-driven, no full GEMM)
 *
 * Sweeps:
 *   - Filter shape: scattered (Bernoulli per row) vs clustered (a few wide
 *     contiguous runs). Clustered fires the direct-range-GEMM dispatch path;
 *     scattered exercises array/sparse-bitmap → gather and dense-bitmap →
 *     range+mask.
 *   - Database size N and selectivity (main grid).
 *   - Dimension D (at fixed N, sel) — covers 128 → 1536-d embeddings.
 *   - Query batch Q (at fixed N, sel) — covers the GEMV→GEMM transition.
 *
 * Reports per row: container-type census (run/array/bitmap), schedule task
 * breakdown (direct-range vs masked-range vs gather), median + std for both
 * paths, QPS (NQ / median * 1000), speedup, recall vs the cuVS reference.
 * 3 outer replicates × ITERS interleaved A/B per config.
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
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

enum class Shape { Scattered, Clustered };
static const char* shape_name(Shape s)
{
  return s == Shape::Scattered ? "scattered" : "clustered";
}

struct Stats { double median, mean, std_dev; };
static Stats stats_of(std::vector<double> t)
{
  if (t.empty()) return {0,0,0};
  std::sort(t.begin(), t.end());
  int n      = static_cast<int>(t.size());
  double sum = 0;
  for (double v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (double v : t) var += (v - mean) * (v - mean);
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

static size_t free_vram() { size_t f=0, t=0; cudaMemGetInfo(&f, &t); return f; }

// Build a filter at the requested selectivity and shape. For clustered,
// produce `n_runs` equally-spaced contiguous runs covering sel * N rows in
// total; each run width = sel * N / n_runs. Wide runs (≥ 64K) trigger the
// direct-range-GEMM dispatch path.
static std::vector<uint32_t> make_filter(int N, double sel, Shape shape,
                                         int n_runs, int seed)
{
  std::vector<uint32_t> ids;
  if (shape == Shape::Scattered) {
    std::mt19937 fg(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    ids.reserve(static_cast<size_t>(N * sel * 1.1));
    for (int i = 0; i < N; ++i)
      if (u(fg) < sel) ids.push_back(static_cast<uint32_t>(i));
  } else {
    int64_t total = static_cast<int64_t>(N * sel);
    if (n_runs < 1) n_runs = 1;
    int64_t width = total / n_runs;
    if (width < 1) width = 1;
    int64_t stride = N / n_runs;
    ids.reserve(static_cast<size_t>(total));
    for (int r = 0; r < n_runs; ++r) {
      int64_t s = static_cast<int64_t>(r) * stride;
      int64_t e = std::min<int64_t>(s + width, N);
      for (int64_t v = s; v < e; ++v) ids.push_back(static_cast<uint32_t>(v));
    }
  }
  return ids;
}

struct TaskBreakdown {
  int      n_direct = 0, n_masked = 0, n_gather = 0;
  uint64_t cols_direct = 0, cols_masked = 0, cols_gather = 0;
};
static TaskBreakdown analyse(const cu_roaring::SearchSchedule& s)
{
  TaskBreakdown b;
  for (const auto& t : s.tasks) {
    if (t.kind == cu_roaring::GemmTask::kGather) {
      b.n_gather++;
      b.cols_gather += t.n_cols;
    } else if (t.mask) {
      b.n_masked++;
      b.cols_masked += t.n_cols;
    } else {
      b.n_direct++;
      b.cols_direct += t.n_cols;
    }
  }
  return b;
}

struct Cfg {
  int   n;
  int   dim;
  int   nq;
  double sel;
  Shape shape;
  int   n_runs;  // for Clustered
};

int main()
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(rng, stream);
  curandSetPseudoRandomGeneratorSeed(rng, 0xC0FFEE);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.1f GB free)\n\n", prop.name,
         prop.multiProcessorCount, free_vram() / (1024.0 * 1024.0 * 1024.0));

  constexpr int K          = 10;
  constexpr int WARMUP     = 10;
  constexpr int ITERS      = 20;   // 30 was overkill; 20 keeps cv < 2%.
  constexpr int REPLICATES = 3;
  constexpr int CLUSTER_RUNS = 4;  // n_runs for the clustered shape

  std::vector<Cfg> cfgs;
  // ---- Main grid: N x sel, both shapes, D=128, Q=64 ----------------------
  const int Ns[]      = {1'000'000, 5'000'000, 10'000'000, 25'000'000, 50'000'000};
  const double sels[] = {0.01, 0.05, 0.10, 0.25, 0.50};
  for (int N : Ns) for (double s : sels) {
    if (N == 50'000'000 && s > 0.10) continue;  // OOM guard (gather)
    cfgs.push_back({N, 128, 64, s, Shape::Scattered, 0});
    cfgs.push_back({N, 128, 64, s, Shape::Clustered, CLUSTER_RUNS});
  }
  // ---- D sweep: N=2M, sel=5%, Q=64 ---------------------------------------
  for (int d : {128, 384, 768, 1536}) {
    cfgs.push_back({2'000'000, d, 64, 0.05, Shape::Scattered, 0});
    cfgs.push_back({2'000'000, d, 64, 0.05, Shape::Clustered, CLUSTER_RUNS});
  }
  // ---- Q sweep: N=10M, sel=5%, D=128 -------------------------------------
  for (int q : {1, 8, 32, 128}) {
    cfgs.push_back({10'000'000, 128, q, 0.05, Shape::Scattered, 0});
    cfgs.push_back({10'000'000, 128, q, 0.05, Shape::Clustered, CLUSTER_RUNS});
  }

  FILE* jf = fopen("bench_schedule_driven_roaring.json", "w");
  fprintf(jf, "{\n  \"gpu\": \"%s\", \"k\": %d,\n", prop.name, K);
  fprintf(jf, "  \"warmup\": %d, \"iters\": %d, \"replicates\": %d,\n",
          WARMUP, ITERS, REPLICATES);
  fprintf(jf, "  \"a\": \"cuvs brute_force + raft bitset_filter\",\n");
  fprintf(jf, "  \"b\": \"cu_roaring schedule-driven search\",\n  \"results\": [\n");

  int prev_n = -1, prev_dim = -1;
  raft::device_matrix<float, int64_t> dataset =
    raft::make_device_matrix<float, int64_t>(res, 0, 0);
  std::unique_ptr<cuvs::neighbors::brute_force::index<float, float>> index;

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  bool first_result = true;

  for (size_t ci = 0; ci < cfgs.size(); ++ci) {
    Cfg c = cfgs[ci];
    int N = c.n, D = c.dim, NQ = c.nq;

    printf("=== N=%d D=%d Q=%d sel=%.0f%% shape=%s ===\n",
           N, D, NQ, c.sel * 100, shape_name(c.shape));
    fflush(stdout);

    // ---- Dataset + cuVS index (rebuilt when (N,D) changes) --------------
    if (N != prev_n || D != prev_dim) {
      index.reset();
      dataset = raft::make_device_matrix<float, int64_t>(res, 0, 0);
      size_t need = static_cast<size_t>(N) * D * sizeof(float) + 2'000'000'000ull;
      if (free_vram() < need) {
        printf("  SKIP: insufficient VRAM\n\n");
        continue;
      }
      dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
      curandGenerateNormal(rng, dataset.data_handle(),
                           static_cast<size_t>(N) * D, 0.0f, 1.0f);
      raft::resource::sync_stream(res);

      cuvs::neighbors::brute_force::index_params bp;
      bp.metric = cuvs::distance::DistanceType::InnerProduct;
      index = std::make_unique<cuvs::neighbors::brute_force::index<float, float>>(
        cuvs::neighbors::brute_force::build(
          res, bp, raft::make_const_mdspan(dataset.view())));
      prev_n = N; prev_dim = D;
    }

    auto queries = raft::make_device_matrix<float, int64_t>(res, NQ, D);
    curandGenerateNormal(rng, queries.data_handle(),
                         static_cast<size_t>(NQ) * D, 0.0f, 1.0f);

    // ---- Filter -----------------------------------------------------------
    std::vector<uint32_t> ids =
      make_filter(N, c.sel, c.shape, c.n_runs, 123 + static_cast<int>(ci));
    int card = static_cast<int>(ids.size());
    double sparsity = 1.0 - static_cast<double>(card) / N;

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

    // For clustered shapes, build the bitmap through CRoaring and
    // run_optimize() so RUN containers survive to the GPU — that's what
    // exercises the direct-range-GEMM dispatch path. upload_from_sorted_ids
    // stores everything as BITMAP containers, defeating that path.
    cu_roaring::GpuRoaring gpu;
    if (c.shape == Shape::Clustered) {
      roaring_bitmap_t* cpu_bm = roaring_bitmap_create();
      int    nr     = c.n_runs > 0 ? c.n_runs : 1;
      int64_t total = static_cast<int64_t>(N * c.sel);
      int64_t width = std::max<int64_t>(1, total / nr);
      int64_t stride = N / nr;
      for (int r = 0; r < nr; ++r) {
        int64_t s = static_cast<int64_t>(r) * stride;
        int64_t e = std::min<int64_t>(s + width, N);
        if (e > s)
          roaring_bitmap_add_range(cpu_bm, static_cast<uint64_t>(s),
                                           static_cast<uint64_t>(e));
      }
      roaring_bitmap_run_optimize(cpu_bm);
      gpu = cu_roaring::upload(cpu_bm, static_cast<uint32_t>(N));
      roaring_bitmap_free(cpu_bm);
    } else {
      gpu = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(card), static_cast<uint32_t>(N));
    }

    size_t gather_bytes = static_cast<size_t>(card) * D * sizeof(float);
    if (free_vram() < gather_bytes + 1'500'000'000ull) {
      printf("  SKIP: gather buffer (%.1f GB) would not fit\n\n", gather_bytes / 1e9);
      cu_roaring::gpu_roaring_free(gpu);
      continue;
    }
    auto sched = cu_roaring::build_schedule(gpu, N, dataset.data_handle(),
                                            D, stream);
    TaskBreakdown tb = analyse(sched);

    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, NQ, K);
    auto dists     = raft::make_device_matrix<float, int64_t>(res, NQ, K);
    cuvs::neighbors::brute_force::search_params sp;
    uint32_t* d_rid = nullptr;
    float*    d_rsc = nullptr;
    cudaMalloc(&d_rid, static_cast<size_t>(NQ) * K * sizeof(uint32_t));
    cudaMalloc(&d_rsc, static_cast<size_t>(NQ) * K * sizeof(float));

    auto run_a = [&] {
      cuvs::neighbors::brute_force::search(res, sp, *index,
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

    std::vector<double> all_a, all_b;
    for (int rep = 0; rep < REPLICATES; ++rep)
      for (int i = 0; i < ITERS; ++i) {
        if (i & 1) { all_b.push_back(time(run_b)); all_a.push_back(time(run_a)); }
        else       { all_a.push_back(time(run_a)); all_b.push_back(time(run_b)); }
      }
    auto sa = stats_of(all_a);
    auto sb = stats_of(all_b);

    // Recall vs cuVS bitset reference.
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

    double qps_a   = NQ / (sa.median * 1e-3);
    double qps_b   = NQ / (sb.median * 1e-3);
    double speedup = sa.median / sb.median;
    printf("  card=%d sparsity=%.4f  containers: run=%u arr=%u bmp=%u\n",
           card, sparsity, gpu.n_run_containers, gpu.n_array_containers,
           gpu.n_bitmap_containers);
    printf("  schedule: direct=%d (%.1fM cols) masked=%d (%.1fM cols) gather=%d (%.1fM cols)\n",
           tb.n_direct, tb.cols_direct / 1e6, tb.n_masked,
           tb.cols_masked / 1e6, tb.n_gather, tb.cols_gather / 1e6);
    printf("  bitset:  median %.3f ms (std %.3f)  %.0f QPS\n",
           sa.median, sa.std_dev, qps_a);
    printf("  roaring: median %.3f ms (std %.3f)  %.0f QPS\n",
           sb.median, sb.std_dev, qps_b);
    printf("  speedup: %.2fx   recall@%d: %.4f   fallback=%d\n\n",
           speedup, K, recall, sched.used_fallback ? 1 : 0);
    fflush(stdout);

    fprintf(jf, "    %s{\"n\":%d,\"dim\":%d,\"nq\":%d,\"sel\":%.3f,"
                "\"shape\":\"%s\",\"n_runs\":%d,\"card\":%d,\"sparsity\":%.4f,"
                "\"containers\":{\"run\":%u,\"array\":%u,\"bitmap\":%u},"
                "\"schedule\":{\"direct_tasks\":%d,\"direct_cols\":%llu,"
                "\"masked_tasks\":%d,\"masked_cols\":%llu,"
                "\"gather_tasks\":%d,\"gather_cols\":%llu},"
                "\"bitset_ms\":%.4f,\"bitset_std\":%.4f,\"bitset_qps\":%.0f,"
                "\"roaring_ms\":%.4f,\"roaring_std\":%.4f,\"roaring_qps\":%.0f,"
                "\"speedup\":%.4f,\"recall\":%.4f,\"fallback\":%d}",
            first_result ? "" : ",\n",
            N, D, NQ, c.sel, shape_name(c.shape), c.n_runs, card, sparsity,
            gpu.n_run_containers, gpu.n_array_containers, gpu.n_bitmap_containers,
            tb.n_direct, (unsigned long long)tb.cols_direct,
            tb.n_masked, (unsigned long long)tb.cols_masked,
            tb.n_gather, (unsigned long long)tb.cols_gather,
            sa.median, sa.std_dev, qps_a, sb.median, sb.std_dev, qps_b,
            speedup, recall, sched.used_fallback ? 1 : 0);
    first_result = false;

    cudaFree(d_rid);
    cudaFree(d_rsc);
    cu_roaring::free_schedule(sched);
    cu_roaring::gpu_roaring_free(gpu);
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  curandDestroyGenerator(rng);
  cublasDestroy(cublas);
  printf("wrote bench_schedule_driven_roaring.json\n");
  return 0;
}
