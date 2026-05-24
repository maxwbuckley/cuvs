# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
#
# Roaring-bitmap-filtered ANN search benchmarks.
#
# This subdirectory is added by cuVS's top-level CMakeLists when
# CUVS_BUILD_ROARING_BENCH=ON. All targets here link against:
#   - cuvs                  (the main library; provides brute_force / IVF /
#                            CAGRA + roaring_filter)
#   - cu_roaring_bitmap     (fetched by cmake/thirdparty/get_cu_roaring.cmake)
#   - raft::raft, rmm::rmm  (cuvs's own transitive deps)
#   - CUDA::cudart, CUDA::cublas, CUDA::curand
#
# There are no machine-local paths; a clean clone of cuVS plus
# `cmake -DCUVS_BUILD_ROARING_BENCH=ON ..` should be enough.

set(BENCH_CUDA_FLAGS --extended-lambda --expt-relaxed-constexpr)
set(BENCH_COMPILE_DEFS
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)

# Helper: standard cuvs-bench-target wiring. Single TU, links against the
# in-build cuvs lib + cu_roaring + standard runtime deps.
function(cuvs_add_roaring_bench target src)
    add_executable(${target} ${src})
    target_link_libraries(${target}
        PRIVATE
            cuvs
            cu_roaring_bitmap
            raft::raft
            rmm::rmm
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
    )
    target_compile_definitions(${target} PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD              17
        CUDA_STANDARD_REQUIRED     ON
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>
    )
endfunction()

# ---------------------------------------------------------------------------
# End-to-end selectivity sweep — cuVS bitset_filter vs cuVS roaring_filter
# (SDDMM / warp_contains / schedule-driven, switched via CUVS_ROARING_DISPATCH).
# Environment knobs: E2E_DTYPE=fp16, E2E_SHAPE=clustered, E2E_SELS=...,
#                    CUVS_ROARING_DISPATCH=schedule
cuvs_add_roaring_bench(bench_e2e_sweep bench_e2e_sweep.cu)

# Real-data YFCC-10M sweep — per-query tag-tuple intersection, host vs gpu
# build, RUN-aware container census.
cuvs_add_roaring_bench(bench_yfcc_search bench_yfcc_search.cu)

# Standalone exercise of cu_roaring's schedule-driven path (sanity bench;
# does not go through cuvs::brute_force::search dispatch).
cuvs_add_roaring_bench(bench_schedule_driven_roaring bench_schedule_driven_roaring.cu)

# Synthetic-generator sweep: load roaring .bin filters produced by the
# roaring-benchmark generator suite. Set SYNTH_FILTER_DIR=/path at run time.
cuvs_add_roaring_bench(bench_synth_filters bench_synth_filters.cu)

# Brute-force-specific bench (filter shapes against cuvs::brute_force::search).
cuvs_add_roaring_bench(bench_brute_force_roaring bench_brute_force_roaring.cu)

# IVF-Flat bench against the new roaring interleaved-scan instantiation.
cuvs_add_roaring_bench(bench_ivf_flat_roaring bench_ivf_flat_roaring.cu)

# CAGRA bench (refreshed; the _comprehensive bench is data-driven).
cuvs_add_roaring_bench(bench_cagra_roaring bench_cagra_roaring.cu)
cuvs_add_roaring_bench(bench_cagra_roaring_comprehensive bench_cagra_roaring_comprehensive.cu)

# Micro-benches.
cuvs_add_roaring_bench(bench_throughput bench_throughput.cu)
cuvs_add_roaring_bench(bench_transfer  bench_transfer.cu)

# Nsight-Compute-friendly entry for profiling CAGRA roaring search.
cuvs_add_roaring_bench(profile_cagra_roaring profile_cagra_roaring.cu)
