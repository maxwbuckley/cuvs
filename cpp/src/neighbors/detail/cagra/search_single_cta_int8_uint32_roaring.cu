/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_filter_utils.cuh"
#include "search_single_cta_inst.cuh"

#include <cuvs/neighbors/roaring_filter.cuh>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

instantiate_kernel_selection(
  int8_t,
  uint32_t,
  float,
  CagraSampleFilterWithQueryIdOffset<cuvs::neighbors::filtering::roaring_filter>);

instantiate_kernel_selection(
  int8_t,
  uint32_t,
  float,
  CagraSampleFilterWithQueryIdOffset<cuvs::neighbors::filtering::roaring_filter_warp>);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
