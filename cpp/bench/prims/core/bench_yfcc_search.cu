/*
 * Benchmark: schedule-driven vs cuVS bitset_filter on real YFCC-10M queries.
 *
 * For each sampled query we build the per-query filter from its tag set
 * (intersection of tag bitmaps), upload it to the GPU, time a Q=1 top-10
 * search with each backend, and average. Per-query filter construction is
 * inside the timed region because YFCC is a per-query-filter workload — the
 * "schedule reused across batch" assumption does not apply here.
 *
 *   YFCC_DATA=/path/to/yfcc_data        # has queries.bin + tags/
 *   YFCC_VEC=/path/to/base.10M.u8bin    # Big-ANN u8bin
 *   YFCC_SAMPLE=256                     # queries to time
 *   YFCC_TAG_DIR=/path/to/tag_dir       # overrides YFCC_DATA/tags
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring&);
void gpu_roaring_free_async(GpuRoaring&, cudaStream_t);
}

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

// Big-ANN u8bin header: [int32 nvecs, int32 dim] then nvecs*dim uint8.
static std::vector<uint8_t> load_u8bin(const std::string& path, int* n, int* d)
{
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open %s\n", path.c_str()); std::exit(1); }
  int32_t hdr[2];
  fread(hdr, sizeof(int32_t), 2, f);
  *n = hdr[0]; *d = hdr[1];
  std::vector<uint8_t> buf(static_cast<size_t>(*n) * *d);
  fread(buf.data(), 1, buf.size(), f);
  fclose(f);
  return buf;
}

// Per-query tag tuples. queries.bin = [uint32 n_q][repeat n_q: uint32 n_t,
// n_t * uint32 tag_ids].
static std::vector<std::vector<uint32_t>> load_query_tags(const std::string& path)
{
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open %s\n", path.c_str()); std::exit(1); }
  uint32_t nq = 0;
  fread(&nq, sizeof(uint32_t), 1, f);
  std::vector<std::vector<uint32_t>> qt(nq);
  for (uint32_t q = 0; q < nq; ++q) {
    uint32_t nt = 0;
    fread(&nt, sizeof(uint32_t), 1, f);
    qt[q].resize(nt);
    if (nt) fread(qt[q].data(), sizeof(uint32_t), nt, f);
  }
  fclose(f);
  return qt;
}

// tag_<id>.bin = [uint32 n_ids, uint32 tag_id] then n_ids * uint32 ids (sorted).
static roaring_bitmap_t* load_tag_bitmap(const std::string& path, uint32_t N)
{
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return nullptr;
  uint32_t hdr[2];
  fread(hdr, sizeof(uint32_t), 2, f);
  std::vector<uint32_t> ids(hdr[0]);
  if (hdr[0]) fread(ids.data(), sizeof(uint32_t), hdr[0], f);
  fclose(f);
  // Trim to [0, N) just in case the source universe differs.
  while (!ids.empty() && ids.back() >= N) ids.pop_back();
  return roaring_bitmap_of_ptr(ids.size(), ids.data());
}

int main()
{
  const char* env_data    = std::getenv("YFCC_DATA");
  const char* env_vec     = std::getenv("YFCC_VEC");
  const char* env_sample  = std::getenv("YFCC_SAMPLE");
  const char* env_tag_dir = std::getenv("YFCC_TAG_DIR");
  if (!env_data || !env_vec) {
    fprintf(stderr, "set YFCC_DATA=/.../yfcc_data and "
                    "YFCC_VEC=/.../base.10M.u8bin\n");
    return 1;
  }
  std::string data_dir(env_data);
  std::string vec_path(env_vec);
  std::string tag_dir = env_tag_dir ? env_tag_dir : data_dir + "/tags";
  int sample = env_sample ? std::atoi(env_sample) : 256;
  constexpr int K = 10;

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  cublasHandle_t cublas;
  cublasCreate(&cublas);

  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);

  // ---- Load base vectors ----------------------------------------------------
  int N = 0, D = 0;
  printf("loading %s ...\n", vec_path.c_str()); fflush(stdout);
  auto h_u8 = load_u8bin(vec_path, &N, &D);
  printf("  base: N=%d D=%d (%.2f GB u8)\n", N, D, h_u8.size() / 1e9);
  std::vector<float> h_f(static_cast<size_t>(N) * D);
  for (size_t i = 0; i < h_f.size(); ++i) h_f[i] = static_cast<float>(h_u8[i]);

  auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
  raft::update_device(dataset.data_handle(), h_f.data(), h_f.size(), stream);
  raft::resource::sync_stream(res);
  h_f.clear(); h_f.shrink_to_fit();
  h_u8.clear(); h_u8.shrink_to_fit();
  printf("  uploaded base float (%.2f GB)\n", static_cast<double>(N) * D * 4 / 1e9);

  cuvs::neighbors::brute_force::index_params bp;
  bp.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index = cuvs::neighbors::brute_force::build(
    res, bp, raft::make_const_mdspan(dataset.view()));

  // ---- Load query vectors (Big-ANN u8bin alongside base) -------------------
  std::string qvec = vec_path;
  auto slash = qvec.find_last_of('/');
  qvec = (slash == std::string::npos ? std::string() : qvec.substr(0, slash + 1))
       + "query.public.100K.u8bin";
  int NQ_all = 0, Dq = 0;
  auto h_qu8 = load_u8bin(qvec, &NQ_all, &Dq);
  if (Dq != D) {
    fprintf(stderr, "query dim %d != base dim %d\n", Dq, D);
    return 1;
  }
  printf("  queries: NQ=%d D=%d\n", NQ_all, D);
  std::vector<float> h_qf(static_cast<size_t>(NQ_all) * D);
  for (size_t i = 0; i < h_qf.size(); ++i) h_qf[i] = static_cast<float>(h_qu8[i]);

  // ---- Load per-query tag tuples + the unique tag set ---------------------
  std::string qpath = data_dir + "/queries.bin";
  auto qtags = load_query_tags(qpath);
  printf("  loaded %zu query tag tuples\n", qtags.size());

  std::vector<uint32_t> q_indices;  // global query ids we'll time
  q_indices.reserve(sample);
  // Sample evenly across the 100K queries for variety.
  for (int i = 0; i < sample && i < static_cast<int>(qtags.size()); ++i)
    q_indices.push_back(static_cast<uint32_t>(i * (qtags.size() / sample)));

  std::unordered_map<uint32_t, roaring_bitmap_t*> tag_cache;
  auto get_tag = [&](uint32_t tag) -> roaring_bitmap_t* {
    auto it = tag_cache.find(tag);
    if (it != tag_cache.end()) return it->second;
    char p[512];
    std::snprintf(p, sizeof(p), "%s/tag_%u.bin", tag_dir.c_str(), tag);
    roaring_bitmap_t* bm = load_tag_bitmap(p, N);
    tag_cache.emplace(tag, bm);
    return bm;
  };

  // ---- Per-query timing ----------------------------------------------------
  uint32_t* d_rid = nullptr;
  float*    d_rsc = nullptr;
  cudaMalloc(&d_rid, K * sizeof(uint32_t));
  cudaMalloc(&d_rsc, K * sizeof(float));
  auto q_dev = raft::make_device_matrix<float, int64_t>(res, 1, D);

  std::vector<double> ta_search, tb_search;        // search only (warmed)
  std::vector<double> tb_build,  tb_endtoend;      // host-path schedule build, end-to-end
  std::vector<double> tb_build_gpu, tb_endtoend_gpu; // GPU-path schedule build, end-to-end
  std::vector<uint32_t> cards;
  // Container-type + task-type histograms, summed across all sampled queries.
  // These tell us whether sorting + run_optimize actually produces RUN-dominant
  // schedules (sorted YFCC) or whether the filter degenerates to bitmap/array
  // containers (unsorted YFCC).
  uint64_t sum_run = 0, sum_arr = 0, sum_bmp = 0;
  uint64_t sum_direct = 0, sum_masked = 0, sum_gather = 0;
  uint64_t sum_direct_cols = 0, sum_masked_cols = 0, sum_gather_cols = 0;
  uint32_t q_run_only = 0;     // queries whose schedule is 100% RUN containers
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);

  size_t skipped = 0;
  double recall = 0.0;   // last query's recall (gpu-bitset path vs cuVS bitset)
  for (uint32_t qi : q_indices) {
    // Per-query filter: intersection of tag bitmaps.
    roaring_bitmap_t* filt = nullptr;
    for (uint32_t tag : qtags[qi]) {
      roaring_bitmap_t* t = get_tag(tag);
      if (!t) { if (filt) { roaring_bitmap_free(filt); filt = nullptr; } break; }
      if (!filt)  filt = roaring_bitmap_copy(t);
      else        roaring_bitmap_and_inplace(filt, t);
    }
    if (!filt || roaring_bitmap_get_cardinality(filt) == 0) {
      if (filt) roaring_bitmap_free(filt);
      ++skipped;
      continue;
    }
    // Run-optimise so the GPU schedule sees RUN containers when the data is
    // sorted (no-op on unsorted/scattered layouts).
    roaring_bitmap_run_optimize(filt);

    uint64_t card = roaring_bitmap_get_cardinality(filt);
    cards.push_back(static_cast<uint32_t>(card));

    // Pre-pack the host bitset words ONCE (the iterate-CRoaring-and-OR-bits
    // step). This is host work that any cuVS user has to do somehow to
    // materialise a bitset on the host before H2D. Doing it outside the
    // timed lambda lets us reuse the same packed buffer across iters; the
    // timed region still pays the H2D, the bitset device-alloc, and the
    // bitset_filter wrap on every call.
    uint32_t nw = (static_cast<uint32_t>(N) + 31) / 32;
    std::vector<uint32_t> hb(nw, 0);
    {
      roaring_uint32_iterator_t* it = roaring_iterator_create(filt);
      while (it->has_value) {
        hb[it->current_value / 32] |= (1u << (it->current_value % 32));
        roaring_uint32_iterator_advance(it);
      }
      roaring_uint32_iterator_free(it);
    }

    // upload query vector
    raft::update_device(q_dev.data_handle(),
                        h_qf.data() + static_cast<size_t>(qi) * D,
                        D, stream);

    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, K);
    auto dists     = raft::make_device_matrix<float, int64_t>(res, 1, K);
    cuvs::neighbors::brute_force::search_params sp;

    // run_a (cuVS) — the production-realistic per-query path. Build the
    // device bitset (alloc + zero + H2D) inside the timed region, since
    // every cuVS bitset_filter user has to pay this somehow.
    auto run_a = [&] {
      raft::core::bitset<uint32_t, int64_t> bits(res, N, false);
      raft::update_device(bits.data(), hb.data(), nw, stream);
      auto bitset_filt =
        cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bits.view());
      cuvs::neighbors::brute_force::search(res, sp, index,
        raft::make_const_mdspan(q_dev.view()),
        neighbors.view(), dists.view(), bitset_filt);
    };

    // ---- B: cu_roaring (host CRoaring -> GPU, preserves RUN containers) ----
    // The production-realistic path. The filter is already on host (it had
    // to be — CRoaring intersection of tag bitmaps is host work). Take it
    // straight to GPU via upload(filt, N, stream), build the schedule on
    // top, search. This preserves the run_optimize info; the alternative
    // upload_from_device_bitset path silently flattens everything to bitmap
    // containers and we have no reason to round-trip through a flat bitset.
    cudaEventRecord(e0, stream);
    auto gpu_host_path = cu_roaring::upload(filt, static_cast<uint32_t>(N),
                                            stream);
    auto sched_host    = cu_roaring::build_schedule(gpu_host_path, N,
                                                    dataset.data_handle(),
                                                    D, stream);
    cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
    float ms_build = 0; cudaEventElapsedTime(&ms_build, e0, e1);

    // Container + task-type census for the (sole) schedule.
    sum_run += gpu_host_path.n_run_containers;
    sum_arr += gpu_host_path.n_array_containers;
    sum_bmp += gpu_host_path.n_bitmap_containers;
    bool all_run = (gpu_host_path.n_array_containers == 0u &&
                    gpu_host_path.n_bitmap_containers == 0u &&
                    gpu_host_path.n_run_containers > 0u);
    int n_dir = 0, n_msk = 0, n_gth = 0;
    uint64_t c_dir = 0, c_msk = 0, c_gth = 0;
    for (const auto& t : sched_host.tasks) {
      if (t.kind == cu_roaring::GemmTask::kGather) { ++n_gth; c_gth += t.n_cols; }
      else if (t.mask)                              { ++n_msk; c_msk += t.n_cols; }
      else                                          { ++n_dir; c_dir += t.n_cols; }
    }
    sum_direct += n_dir; sum_masked += n_msk; sum_gather += n_gth;
    sum_direct_cols += c_dir; sum_masked_cols += c_msk; sum_gather_cols += c_gth;
    if (all_run && n_msk == 0 && n_gth == 0) ++q_run_only;

    // Also keep the gpu-bitset BUILD measurement as a reference point — what
    // it would cost if someone already had a device bitset (e.g. from a GPU
    // OR-fold) and didn't want to round-trip through CRoaring. Build it
    // once for timing; not used for the search loop.
    raft::core::bitset<uint32_t, int64_t> bits_ref(res, N, false);
    raft::update_device(bits_ref.data(), hb.data(), nw, stream);
    raft::resource::sync_stream(res);
    cudaEventRecord(e0, stream);
    auto gpu_dev_path = cu_roaring::upload_from_device_bitset(
        reinterpret_cast<const uint32_t*>(bits_ref.data()), nw,
        static_cast<uint32_t>(N), stream);
    cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
    float ms_build_gpu = 0;
    cudaEventElapsedTime(&ms_build_gpu, e0, e1);
    cu_roaring::gpu_roaring_free_async(gpu_dev_path, stream);

    // Use the host-path schedule for the timed search loop — this is the
    // path that sees RUN containers when the input is sorted+run_optimize'd.
    auto& sched = sched_host;
    auto run_b = [&] {
      cu_roaring::roaring_filtered_search(cublas, q_dev.data_handle(), 1,
        dataset.data_handle(), N, D, sched, K, d_rid, d_rsc, stream);
    };

    // Warm both paths' steady state (graph capture for B, kernel cache for A).
    for (int i = 0; i < 4; ++i) { run_a(); run_b(); }
    cudaStreamSynchronize(stream);

    auto time = [&](auto&& f) {
      cudaEventRecord(e0, stream); f();
      cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); return static_cast<double>(ms);
    };

    constexpr int ITERS = 16;
    double s_a = 0, s_b = 0;
    for (int i = 0; i < ITERS; ++i) {
      if (i & 1) { s_b += time(run_b); s_a += time(run_a); }
      else       { s_a += time(run_a); s_b += time(run_b); }
    }
    double ma = s_a / ITERS, mb = s_b / ITERS;
    ta_search.push_back(ma);
    tb_search.push_back(mb);
    tb_build.push_back(static_cast<double>(ms_build));
    tb_endtoend.push_back(static_cast<double>(ms_build) + mb);
    tb_build_gpu.push_back(static_cast<double>(ms_build_gpu));
    tb_endtoend_gpu.push_back(static_cast<double>(ms_build_gpu) + mb);

    // Recall check: re-run both paths once, compare returned ids.
    run_a();
    std::vector<int64_t> ref(K);
    raft::update_host(ref.data(), neighbors.data_handle(), K, stream);
    raft::resource::sync_stream(res);
    run_b();
    cudaStreamSynchronize(stream);
    std::vector<uint32_t> got(K);
    cudaMemcpy(got.data(), d_rid, K * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    recall = recall_at_k(got, ref, 1, K);

    cu_roaring::free_schedule(sched_host);
    cu_roaring::gpu_roaring_free_async(gpu_host_path, stream);
    roaring_bitmap_free(filt);
  }

  auto sa  = stats_of(ta_search);
  auto sbs = stats_of(tb_search);
  auto sbb = stats_of(tb_build);
  auto sbe = stats_of(tb_endtoend);
  auto sbbg = stats_of(tb_build_gpu);
  auto sbeg = stats_of(tb_endtoend_gpu);
  auto sc  = stats_of(std::vector<double>(cards.begin(), cards.end()));

  const char* env_label = std::getenv("YFCC_LABEL");
  std::string label = env_label ? env_label : "yfcc";
  size_t timed = ta_search.size();
  printf("\nYFCC[%s]: timed %zu / %d queries (skipped %zu with empty filter)\n",
         label.c_str(), timed, sample, skipped);
  printf("  card per query:    median %.0f, mean %.0f\n", sc.median, sc.mean);
  if (timed > 0) {
    double q = static_cast<double>(timed);
    printf("  schedule path:     host CRoaring (run_optimize) -> upload(filt, N)\n");
    printf("  containers/query:  run=%.1f arr=%.1f bmp=%.1f   (%u/%zu queries are 100%% RUN)\n",
           sum_run / q, sum_arr / q, sum_bmp / q, q_run_only, timed);
    printf("  schedule/query:    direct=%.1f (%.1fM cols)  masked=%.1f (%.1fM cols)  gather=%.1f (%.1fM cols)\n",
           sum_direct / q, sum_direct_cols / q / 1e6,
           sum_masked / q, sum_masked_cols / q / 1e6,
           sum_gather / q, sum_gather_cols / q / 1e6);
  }
  printf("  cuVS bitset:       median %.3f ms  (mean %.3f)  → %.0f QPS\n",
         sa.median, sa.mean, 1000.0 / sa.median);
  printf("  roaring SEARCH:    median %.3f ms  (mean %.3f)  → %.0f QPS\n",
         sbs.median, sbs.mean, 1000.0 / sbs.median);
  printf("  roaring BUILD:     median %.3f ms  (mean %.3f)  one-time per query\n",
         sbb.median, sbb.mean);
  printf("  roaring END-TO-END (build + search): median %.3f ms  → %.0f QPS\n",
         sbe.median, 1000.0 / sbe.median);
  printf("  roaring BUILD (gpu-bitset path):     median %.3f ms\n", sbbg.median);
  printf("  roaring END-TO-END  (gpu-bitset):    median %.3f ms  → %.0f QPS\n",
         sbeg.median, 1000.0 / sbeg.median);
  printf("  speedup search-only      : %.2fx\n", sa.median / sbs.median);
  printf("  speedup end-to-end (host): %.2fx\n", sa.median / sbe.median);
  printf("  speedup end-to-end (gpu) : %.2fx\n", sa.median / sbeg.median);
  printf("  recall@%d (last query)   : %.4f\n", K, recall);

  FILE* jf = fopen("bench_yfcc_search.json", "w");
  fprintf(jf, "{\n  \"gpu\":\"%s\",\"n\":%d,\"dim\":%d,\"k\":%d,\n",
          prop.name, N, D, K);
  fprintf(jf, "  \"sampled_queries\":%zu,\"skipped_empty\":%zu,\n",
          ta_search.size(), skipped);
  fprintf(jf, "  \"card_median\":%.0f,\"card_mean\":%.1f,\n", sc.median, sc.mean);
  fprintf(jf, "  \"cuvs_bitset_ms_median\":%.4f,\"cuvs_bitset_ms_mean\":%.4f,\n",
          sa.median, sa.mean);
  fprintf(jf, "  \"roaring_search_ms_median\":%.4f,\"roaring_search_ms_mean\":%.4f,\n",
          sbs.median, sbs.mean);
  fprintf(jf, "  \"roaring_build_ms_median\":%.4f,\"roaring_build_ms_mean\":%.4f,\n",
          sbb.median, sbb.mean);
  fprintf(jf, "  \"roaring_endtoend_ms_median\":%.4f,\"roaring_endtoend_ms_mean\":%.4f,\n",
          sbe.median, sbe.mean);
  fprintf(jf, "  \"roaring_build_gpu_ms_median\":%.4f,\"roaring_build_gpu_ms_mean\":%.4f,\n",
          sbbg.median, sbbg.mean);
  fprintf(jf, "  \"roaring_endtoend_gpu_ms_median\":%.4f,\"roaring_endtoend_gpu_ms_mean\":%.4f,\n",
          sbeg.median, sbeg.mean);
  fprintf(jf, "  \"recall_at_k\":%.4f,\n", recall);
  fprintf(jf, "  \"speedup_search_only\":%.4f,\n", sa.median / sbs.median);
  fprintf(jf, "  \"speedup_endtoend_host\":%.4f,\n", sa.median / sbe.median);
  fprintf(jf, "  \"speedup_endtoend_gpu\":%.4f,\n", sa.median / sbeg.median);
  if (timed > 0) {
    double q = static_cast<double>(timed);
    fprintf(jf, "  \"containers_per_q\":{\"run\":%.2f,\"array\":%.2f,\"bitmap\":%.2f},\n",
            sum_run / q, sum_arr / q, sum_bmp / q);
    fprintf(jf, "  \"tasks_per_q\":{\"direct\":%.2f,\"masked\":%.2f,\"gather\":%.2f},\n",
            sum_direct / q, sum_masked / q, sum_gather / q);
    fprintf(jf, "  \"cols_per_q\":{\"direct\":%.1f,\"masked\":%.1f,\"gather\":%.1f},\n",
            sum_direct_cols / q, sum_masked_cols / q, sum_gather_cols / q);
    fprintf(jf, "  \"queries_pure_run\":%u,\n", q_run_only);
  }
  fprintf(jf, "  \"label\":\"%s\"\n", label.c_str());
  fprintf(jf, "}\n");
  fclose(jf);
  printf("wrote bench_yfcc_search.json\n");

  for (auto& kv : tag_cache) if (kv.second) roaring_bitmap_free(kv.second);
  cudaFree(d_rid); cudaFree(d_rsc);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cublasDestroy(cublas);
  return 0;
}
