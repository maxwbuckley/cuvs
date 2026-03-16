/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Template instantiation for CAGRA multi-CTA search with roaring_filter.
 */

#include "sample_filter_utils.cuh"
#include "search_multi_cta_inst.cuh"

#include <cuvs/neighbors/roaring_filter.cuh>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

instantiate_kernel_selection(
  float,
  uint32_t,
  float,
  CagraSampleFilterWithQueryIdOffset<cuvs::neighbors::filtering::roaring_filter>);

instantiate_kernel_selection(
  float,
  uint32_t,
  float,
  CagraSampleFilterWithQueryIdOffset<cuvs::neighbors::filtering::roaring_filter_warp>);

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
