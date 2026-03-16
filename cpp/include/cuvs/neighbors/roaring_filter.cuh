/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A filter backed by a GPU Roaring bitmap for cuVS CAGRA search.
 * Performs compressed Roaring membership checks directly during
 * graph traversal, without decompressing to a flat bitset.
 */

#pragma once
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/roaring_warp_query.cuh>
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::filtering {

/**
 * @brief Filter backed by a GPU Roaring bitmap.
 *
 * Point-query implementation: each candidate check performs a binary
 * search over the container key array, then a container-type-specific
 * membership test. Works with CAGRA's SampleFilterT template interface.
 */
struct roaring_filter : public base_filter {
  cu_roaring::GpuRoaringView view_;
  uint32_t cardinality_;  // number of set bits (IDs that pass the filter)
  uint32_t n_rows_;       // total universe size

  roaring_filter(cu_roaring::GpuRoaringView view, uint32_t cardinality, uint32_t n_rows)
    : view_(view), cardinality_(cardinality), n_rows_(n_rows)
  {
  }

  __device__ __forceinline__ bool operator()(uint32_t /*query_idx*/,
                                             uint32_t sample_idx) const
  {
    return view_.contains(sample_idx);
  }

  FilterType get_filter_type() const override { return FilterType::Bitset; }
};

/**
 * @brief Warp-cooperative variant of roaring_filter.
 *
 * Threads in the same warp that query IDs with the same high-16 key
 * share the binary search result via __match_any_sync, reducing
 * redundant key lookups during CAGRA's parallel neighbour expansion.
 */
struct roaring_filter_warp : public base_filter {
  cu_roaring::GpuRoaringView view_;
  uint32_t cardinality_;
  uint32_t n_rows_;

  roaring_filter_warp(cu_roaring::GpuRoaringView view, uint32_t cardinality, uint32_t n_rows)
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

}  // namespace cuvs::neighbors::filtering
