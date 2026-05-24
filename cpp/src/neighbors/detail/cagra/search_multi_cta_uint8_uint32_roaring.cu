/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sample_filter_utils.cuh"
#include "search_multi_cta_inst.cuh"

#include <cuvs/neighbors/roaring_filter.cuh>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

instantiate_kernel_selection(
  uint8_t,
  uint32_t,
  float,
  CagraSampleFilterWithQueryIdOffset<cuvs::neighbors::filtering::roaring_filter>);

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
