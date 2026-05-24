# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
#
# Pull in cu_roaring_bitmap as a build-time dependency. The library is
# fetched via FetchContent from the run-optimizations branch of
# maxwbuckley/cu-roaring-bitmap; pin to a specific commit for
# reproducibility.
#
# Overrides:
#   CU_ROARING_SOURCE_DIR  — local checkout (skip git fetch)
#   CU_ROARING_GIT_TAG     — commit/branch to fetch (default below)
#
# Targets exposed: `cu_roaring_bitmap` (static lib), `cu_roaring_device`
# (interface-only headers).

set(CU_ROARING_GIT_TAG  "run-optimizations"
    CACHE STRING "cu_roaring commit/branch/tag")
set(CU_ROARING_REPO_URL "https://github.com/maxwbuckley/cu-roaring-bitmap.git"
    CACHE STRING "cu_roaring git URL")
set(CU_ROARING_SOURCE_DIR ""
    CACHE PATH   "Local cu_roaring checkout (overrides git fetch)")

function(find_and_configure_cu_roaring)
    # Disable cu_roaring's own tests / benches / v2 inside the cuVS build
    # (we just want the library targets).
    set(CU_ROARING_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(CU_ROARING_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(CU_ROARING_BUILD_V2         OFF CACHE BOOL "" FORCE)

    include(FetchContent)
    if(CU_ROARING_SOURCE_DIR)
        message(STATUS "cuVS: using local cu_roaring at ${CU_ROARING_SOURCE_DIR}")
        FetchContent_Declare(cu_roaring SOURCE_DIR "${CU_ROARING_SOURCE_DIR}")
    else()
        message(STATUS "cuVS: fetching cu_roaring from ${CU_ROARING_REPO_URL}@${CU_ROARING_GIT_TAG}")
        FetchContent_Declare(
            cu_roaring
            GIT_REPOSITORY "${CU_ROARING_REPO_URL}"
            GIT_TAG        "${CU_ROARING_GIT_TAG}"
            GIT_SHALLOW    FALSE
            # CRoaring lives as a submodule; pull it together with the parent.
            GIT_SUBMODULES_RECURSE TRUE
        )
    endif()
    FetchContent_MakeAvailable(cu_roaring)
endfunction()

find_and_configure_cu_roaring()
