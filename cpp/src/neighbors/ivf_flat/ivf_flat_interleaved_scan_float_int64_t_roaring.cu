/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit template instantiation of IVF-Flat interleaved scan
 * with roaring_filter for float/int64_t.
 */
#include "ivf_flat_interleaved_scan_explicit_inst.cuh"
#include <cuvs/neighbors/roaring_filter.cuh>

namespace cuvs::neighbors::ivf_flat::detail {

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(float,
                                    int64_t,
                                    cuvs::neighbors::filtering::roaring_filter);

}  // namespace cuvs::neighbors::ivf_flat::detail
