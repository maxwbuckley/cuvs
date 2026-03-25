/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Warp-cooperative Roaring bitmap filter for cuVS CAGRA search.
 * Performs compressed membership checks directly during graph traversal.
 */

#pragma once
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/roaring_warp_query.cuh>
#include <cu_roaring/device/make_view.cuh>
#include <cu_roaring/types.cuh>
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::filtering {

/**
 * @brief Warp-cooperative filter backed by a GPU Roaring bitmap.
 *
 * Threads in the same warp that query IDs with the same high-16 key
 * share the binary search result via __match_any_sync, reducing
 * redundant key lookups during CAGRA's parallel neighbour expansion.
 *
 * Usage (simple — one line from GpuRoaring):
 * @code{.cpp}
 *   auto filter = roaring_filter(gpu_bitmap);
 *   cuvs::neighbors::cagra::search(res, params, index, queries, nb, dist, filter);
 * @endcode
 */
struct roaring_filter : public base_filter {
  cu_roaring::GpuRoaringView view_;
  uint32_t cardinality_;
  uint32_t n_rows_;

  /// Construct directly from a GpuRoaring — no make_view() or cardinality needed.
  explicit roaring_filter(const cu_roaring::GpuRoaring& bitmap)
    : view_(cu_roaring::make_view(bitmap)),
      cardinality_(static_cast<uint32_t>(bitmap.total_cardinality)),
      n_rows_(bitmap.universe_size)
  {
  }

  /// Construct from a pre-built view (advanced usage).
  roaring_filter(cu_roaring::GpuRoaringView view, uint32_t cardinality, uint32_t n_rows)
    : view_(view), cardinality_(cardinality), n_rows_(n_rows)
  {
  }

  __device__ __forceinline__ bool operator()(uint32_t /*query_idx*/,
                                             uint32_t sample_idx) const
  {
    return cu_roaring::warp_contains(view_, sample_idx);
  }

  FilterType get_filter_type() const override { return FilterType::Bitset; }
};

// Keep the old name as an alias for backwards compatibility
using roaring_filter_warp = roaring_filter;

}  // namespace cuvs::neighbors::filtering
