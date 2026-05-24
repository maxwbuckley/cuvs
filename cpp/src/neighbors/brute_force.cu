/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/knn_brute_force.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>
#include <cu_roaring/device/roaring_warp_query.cuh>
#include <cu_roaring/detail/to_csr.cuh>
#include <cu_roaring/detail/filtered_search.cuh>

#include <raft/core/bitset.cuh>
#include <raft/core/copy.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/sparse/linalg/sddmm.hpp>

#include <thrust/transform.h>

#include <cstdlib>
#include <cstring>

namespace cuvs::neighbors::brute_force {

// Brute-force search with roaring_filter.
// - Sparse (pass rate < 10%): decompress roaring → bitset, use CSR+SpGEMM path
// - Dense (pass rate ≥ 10%): use warp_contains() masking in tiled GEMM
namespace roaring_impl {

template <typename T, typename IdxT, typename DistanceT = float>
void brute_force_search_roaring(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistanceT>& idx,
  raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
  const cuvs::neighbors::filtering::roaring_filter& rf,
  raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, IdxT, raft::row_major> distances)
{
  auto metric = idx.metric();

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1), "Value of k must match for outputs");
  RAFT_EXPECTS(idx.dataset().extent(1) == queries.extent(1),
               "Number of columns in queries must match brute force index");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "Only Euclidean, IP, and Cosine are supported!");

  auto stream = raft::resource::get_cuda_stream(res);
  IdxT n_dataset = idx.dataset().extent(0);
  float sparsity = 1.0f - static_cast<float>(rf.cardinality_) / rf.n_rows_;

  // Optional third dispatch: schedule-driven brute_force from cu_roaring's
  // filtered_search.cu. Built via build_schedule(filter, n_rows, db, dim) and
  // executed via roaring_filtered_search / roaring_filtered_search_fp16.
  //
  // The schedule walks the roaring containers and emits per-container GEMM
  // tasks:
  //   RUN containers      -> kRange-direct (one wide GEMM over the slice,
  //                          no gather, no mask; cross-container coalesced)
  //   ARRAY / sparse BMP  -> gathered into a compact buffer, then a Q*card
  //                          skinny GEMM
  //   dense BITMAP        -> kRange-with-mask, masked tile-by-tile in the
  //                          executor (recent coalescing fix merges
  //                          contiguous dense-bitmap tasks up to 1M cols)
  //
  // This is opt-in via CUVS_ROARING_DISPATCH=schedule because the choice
  // between SDDMM (current sparse path) and the schedule depends on filter
  // shape:
  //   - clustered filters with long runs -> schedule wins by a lot (one
  //     direct GEMM beats CSR + masked matmul handily)
  //   - uniform-random sparse filters -> SDDMM and schedule are close
  //   - sel >> 50% -> existing warp_contains path still slightly faster
  //     than the schedule's masked-tile dispatch
  // The right long-term answer is a calibrated auto-dispatch; for now
  // the env var lets the bench harness sweep all three and the user pin a
  // path explicitly.
  const char* dispatch_env = std::getenv("CUVS_ROARING_DISPATCH");
  bool force_schedule = dispatch_env && std::strcmp(dispatch_env, "schedule") == 0;

  if (force_schedule) {
    // Reconstruct a GpuRoaring view-equivalent from rf.view_ (same pattern
    // the SDDMM path below uses).
    cu_roaring::GpuRoaring tmp_roaring{};
    tmp_roaring.keys              = const_cast<uint16_t*>(rf.view_.keys);
    tmp_roaring.types             = reinterpret_cast<cu_roaring::ContainerType*>(
                                      const_cast<cu_roaring::ContainerTypeD*>(rf.view_.types));
    tmp_roaring.offsets           = const_cast<uint32_t*>(rf.view_.offsets);
    tmp_roaring.cardinalities     = const_cast<uint16_t*>(rf.view_.cardinalities);
    tmp_roaring.n_containers      = rf.view_.n_containers;
    tmp_roaring.bitmap_data       = const_cast<uint64_t*>(rf.view_.bitmap_data);
    tmp_roaring.array_data        = const_cast<uint16_t*>(rf.view_.array_data);
    tmp_roaring.run_data          = const_cast<uint16_t*>(rf.view_.run_data);
    tmp_roaring.universe_size     = rf.n_rows_;
    tmp_roaring.total_cardinality = rf.cardinality_;
    tmp_roaring.negated           = rf.view_.negated;

    IdxT n_queries = queries.extent(0);
    IdxT dim       = idx.dataset().extent(1);
    IdxT k         = neighbors.extent(1);

    // build_schedule walks the containers and constructs the task list +
    // gather buffer. cu_roaring uses cublasHandle_t internally; pull it from
    // raft resources (raft maintains a per-stream cuBLAS handle).
    auto cublas = raft::resource::get_cublas_handle(res);
    auto sched  = cu_roaring::build_schedule(tmp_roaring,
                                             static_cast<uint32_t>(n_dataset),
                                             idx.dataset().data_handle(),
                                             static_cast<uint32_t>(dim),
                                             stream);

    // cu_roaring writes top-k as uint32 ids + float scores; cuVS wants IdxT
    // (int64_t) and DistanceT outputs. Allocate a uint32 scratch, copy the
    // float scores through if DistanceT == float, then cast/promote ids.
    rmm::device_uvector<uint32_t> tmp_ids(
        static_cast<size_t>(n_queries) * static_cast<size_t>(k), stream);
    rmm::device_uvector<float> tmp_scores(
        static_cast<size_t>(n_queries) * static_cast<size_t>(k), stream);

    if constexpr (std::is_same_v<T, float>) {
      cu_roaring::roaring_filtered_search(
          cublas, queries.data_handle(), static_cast<uint32_t>(n_queries),
          idx.dataset().data_handle(), static_cast<uint32_t>(n_dataset),
          static_cast<uint32_t>(dim), sched, static_cast<uint32_t>(k),
          tmp_ids.data(), tmp_scores.data(), stream);
    } else if constexpr (std::is_same_v<T, __half>) {
      cu_roaring::roaring_filtered_search_fp16(
          cublas, queries.data_handle(), static_cast<uint32_t>(n_queries),
          idx.dataset().data_handle(), static_cast<uint32_t>(n_dataset),
          static_cast<uint32_t>(dim), sched, static_cast<uint32_t>(k),
          tmp_ids.data(), tmp_scores.data(), stream);
    } else {
      static_assert(std::is_same_v<T, float> || std::is_same_v<T, __half>,
                    "schedule-driven dispatch supports float and __half only");
    }

    // uint32 -> IdxT (int64) and float -> DistanceT.
    thrust::transform(raft::resource::get_thrust_policy(res),
                      tmp_ids.data(), tmp_ids.data() + tmp_ids.size(),
                      neighbors.data_handle(),
                      [] __device__(uint32_t v) { return static_cast<IdxT>(v); });
    if constexpr (std::is_same_v<DistanceT, float>) {
      raft::copy(distances.data_handle(), tmp_scores.data(), tmp_scores.size(), stream);
    } else {
      thrust::transform(raft::resource::get_thrust_policy(res),
                        tmp_scores.data(), tmp_scores.data() + tmp_scores.size(),
                        distances.data_handle(),
                        [] __device__(float v) { return static_cast<DistanceT>(v); });
    }

    cu_roaring::free_schedule(sched);
    return;
  }

  if (sparsity >= 0.9f) {
    // Fused sparse path: roaring → sorted IDs → CSR → SDDMM → epilogue → select_k.
    // Skips bitset intermediate entirely. enumerate_ids produces CSR column indices
    // directly from roaring containers (one kernel, one block per container).

    // Reconstruct a GpuRoaring from the view's device pointers.
    cu_roaring::GpuRoaring tmp_roaring{};
    tmp_roaring.keys              = const_cast<uint16_t*>(rf.view_.keys);
    tmp_roaring.types             = reinterpret_cast<cu_roaring::ContainerType*>(
                                      const_cast<cu_roaring::ContainerTypeD*>(rf.view_.types));
    tmp_roaring.offsets           = const_cast<uint32_t*>(rf.view_.offsets);
    tmp_roaring.cardinalities     = const_cast<uint16_t*>(rf.view_.cardinalities);
    tmp_roaring.n_containers      = rf.view_.n_containers;
    tmp_roaring.bitmap_data       = const_cast<uint64_t*>(rf.view_.bitmap_data);
    tmp_roaring.array_data        = const_cast<uint16_t*>(rf.view_.array_data);
    tmp_roaring.run_data          = const_cast<uint16_t*>(rf.view_.run_data);
    tmp_roaring.universe_size     = rf.n_rows_;
    tmp_roaring.total_cardinality = rf.cardinality_;
    tmp_roaring.negated           = rf.view_.negated;

    IdxT n_queries  = queries.extent(0);
    IdxT dim        = idx.dataset().extent(1);
    IdxT k          = neighbors.extent(1);
    IdxT nnz_per_row = static_cast<IdxT>(rf.cardinality_);
    IdxT total_nnz   = n_queries * nnz_per_row;

    // Step 1: enumerate_ids → sorted column indices for one row
    rmm::device_uvector<int64_t> col_indices_one(nnz_per_row, stream);
    cu_roaring::enumerate_ids(tmp_roaring, col_indices_one.data(), stream);

    // Step 2: build CSR structure
    // indptr: [0, nnz, 2*nnz, ..., n_queries*nnz]
    rmm::device_uvector<IdxT> indptr(n_queries + 1, stream);
    thrust::tabulate(raft::resource::get_thrust_policy(res),
                     indptr.data(), indptr.data() + n_queries + 1,
                     [nnz_per_row] __device__(IdxT i) { return i * nnz_per_row; });

    // Replicate column indices for each query row
    rmm::device_uvector<IdxT> all_indices(total_nnz, stream);
    auto* one_row = col_indices_one.data();
    thrust::for_each_n(raft::resource::get_thrust_policy(res),
                       thrust::make_counting_iterator<IdxT>(0),
                       total_nnz,
                       [one_row, nnz_per_row, out = all_indices.data()] __device__(IdxT i) {
                         out[i] = one_row[i % nnz_per_row];
                       });

    // CSR values (distances — filled by SDDMM)
    rmm::device_uvector<DistanceT> csr_values(total_nnz, stream);

    // Step 3: SDDMM — compute distances at CSR positions only
    auto csr_structure = raft::make_device_compressed_structure_view<IdxT, IdxT, IdxT>(
      indptr.data(), all_indices.data(), n_queries, n_dataset, total_nnz);
    auto csr_view = raft::make_device_csr_matrix_view<DistanceT, IdxT, IdxT, IdxT>(
      csr_values.data(), csr_structure);

    auto dataset_view = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      idx.dataset().data_handle(), n_dataset, dim);

    DistanceT alpha_val = 1.0f;
    DistanceT beta_val  = 0.0f;
    auto alpha = raft::make_host_scalar_view<DistanceT>(&alpha_val);
    auto beta  = raft::make_host_scalar_view<DistanceT>(&beta_val);

    raft::sparse::linalg::sddmm(res,
                                 queries,
                                 dataset_view,
                                 csr_view,
                                 raft::linalg::Operation::NON_TRANSPOSE,
                                 raft::linalg::Operation::TRANSPOSE,
                                 alpha, beta);

    // Step 4: epilogue — apply metric-specific post-processing (L2 norm expansion, etc.)
    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {

      // CSR to COO for row indices (needed by epilogue)
      rmm::device_uvector<IdxT> rows(total_nnz, stream);
      raft::sparse::convert::csr_to_coo(indptr.data(), n_queries, rows.data(), total_nnz, stream);

      // Query norms
      rmm::device_uvector<DistanceT> query_norms(n_queries, stream);
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
          query_norms.data(), queries.data_handle(), dim, n_queries, stream, raft::sqrt_op{});
      } else {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
          query_norms.data(), queries.data_handle(), dim, n_queries, stream, raft::identity_op{});
      }

      ::cuvs::neighbors::detail::epilogue_on_csr(
        res, csr_values.data(), total_nnz, rows.data(), all_indices.data(),
        query_norms.data(), idx.norms().data_handle(), metric);
    }

    // Step 5: select top-k from sparse results
    auto const_csr_view = raft::make_device_csr_matrix_view<const DistanceT, IdxT, IdxT, IdxT>(
      csr_values.data(), csr_structure);
    std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
    bool select_min = cuvs::distance::is_min_close(metric);
    raft::sparse::matrix::select_k(
      res, const_csr_view, no_opt, distances, neighbors, select_min, true);
    return;
  }

  // Dense path: use warp_contains() masking in tiled GEMM
  raft::resources stream_pool_handle(res);
  raft::resource::set_cuda_stream(stream_pool_handle, stream);
  auto idx_norm = idx.has_norms() ? const_cast<DistanceT*>(idx.norms().data_handle()) : nullptr;

  auto view = rf.view_;
  auto mask_fn = [view] __device__(uint32_t id) -> bool {
    return cu_roaring::warp_contains(view, id);
  };

  ::cuvs::neighbors::detail::tiled_brute_force_knn(
    stream_pool_handle,
    queries.data_handle(),
    idx.dataset().data_handle(),
    static_cast<size_t>(queries.extent(0)),
    static_cast<size_t>(idx.dataset().extent(0)),
    static_cast<size_t>(idx.dataset().extent(1)),
    static_cast<size_t>(neighbors.extent(1)),
    distances.data_handle(),
    neighbors.data_handle(),
    metric,
    DistanceT{2.0},
    size_t{0},
    size_t{0},
    idx_norm,
    static_cast<const DistanceT*>(nullptr),
    static_cast<const uint32_t*>(nullptr),
    raft::identity_op(),
    cuvs::neighbors::filtering::FilterType::None,
    mask_fn);
}

}  // namespace roaring_impl

// Roaring-aware search dispatcher: tries roaring_filter first (pointer-based
// dynamic_cast — no exception on failure), then falls through to detail::search.
template <typename T, typename DistT, typename LayoutT>
void search_with_roaring(
  raft::resources const& res,
  const cuvs::neighbors::brute_force::index<T, DistT>& idx,
  raft::device_matrix_view<const T, int64_t, LayoutT> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,
  const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  if constexpr (std::is_same_v<LayoutT, raft::row_major> && std::is_same_v<T, float>) {
    auto* rf = dynamic_cast<const cuvs::neighbors::filtering::roaring_filter*>(&sample_filter);
    if (rf) {
      return roaring_impl::brute_force_search_roaring<T, int64_t, DistT>(
        res, idx, queries, *rf, neighbors, distances);
    }
  }
  cuvs::neighbors::detail::search<T, int64_t, DistT, LayoutT>(
    res, idx, queries, neighbors, distances, sample_filter);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : cuvs::neighbors::index(),
    metric_(cuvs::distance::DistanceType::L2Expanded),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::nullopt),
    metric_arg_(0)
{
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
  raft::resource::sync_stream(res);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
                       std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    dataset_view_(dataset_view),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
                       std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res));
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
index<T, DistT>::index(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
                       std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res));
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_view_ = dataset;
}

template <typename T, typename DistT>
void index<T, DistT>::update_dataset(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_ = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
  raft::copy(res, dataset_.view(), dataset);
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

#define CUVS_INST_BFKNN(T, DistT)                                                               \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)               \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)                 \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,               \
             cuvs::distance::DistanceType metric,                                               \
             DistT metric_arg) -> cuvs::neighbors::brute_force::index<T, DistT>                 \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                           \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::device_matrix_view<const T, int64_t, raft::col_major> dataset)               \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             raft::device_matrix_view<const T, int64_t, raft::col_major> dataset,               \
             cuvs::distance::DistanceType metric,                                               \
             DistT metric_arg) -> cuvs::neighbors::brute_force::index<T, DistT>                 \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                           \
  }                                                                                             \
                                                                                                \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::search_params& params,                        \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    search_with_roaring<T, DistT, raft::row_major>(                                              \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    search_with_roaring<T, DistT, raft::row_major>(                                              \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::search_params& params,                        \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::col_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::col_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
                                                                                                \
  template struct cuvs::neighbors::brute_force::index<T, DistT>;

CUVS_INST_BFKNN(float, float);
CUVS_INST_BFKNN(half, float);

#undef CUVS_INST_BFKNN

}  // namespace cuvs::neighbors::brute_force
