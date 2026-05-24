cmake_minimum_required(VERSION 3.25)
project(roaring_bench LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)

# GCC 13 intrinsics fix for nvcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --pre-include ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/nvcc_pre_include.h")
endif()

# ---------------------------------------------------------------------------
# Paths: local build tree for ABI-matched headers + libraries
# ---------------------------------------------------------------------------
set(CUVS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "cuVS source root")
set(CUVS_BUILD_DIR "${CUVS_ROOT}/build" CACHE PATH "cuVS build directory")
set(DEPS "${CUVS_BUILD_DIR}/_deps")

# Libraries from local build (NOT conda — we need the roaring_filter dispatch)
find_library(CUVS_LIB cuvs HINTS ${CUVS_BUILD_DIR} NO_DEFAULT_PATH)
find_library(RMM_LIB rmm HINTS ${DEPS}/rmm-build NO_DEFAULT_PATH)
message(STATUS "CUVS_LIB: ${CUVS_LIB}")
message(STATUS "RMM_LIB: ${RMM_LIB}")

# cu-roaring-filter standalone library
set(CU_ROARING_DIR "/mnt/c/Users/maxwb/Development/cu-roaring-filter" CACHE PATH "cu-roaring-filter dir")
find_library(CU_ROARING_LIB cu_roaring_filter HINTS ${CU_ROARING_DIR}/build NO_DEFAULT_PATH)
find_library(CROARING_LIB roaring HINTS ${CU_ROARING_DIR}/build/third_party/CRoaring/src NO_DEFAULT_PATH)
message(STATUS "CU_ROARING_LIB: ${CU_ROARING_LIB}")
message(STATUS "CROARING_LIB: ${CROARING_LIB}")

# ---------------------------------------------------------------------------
# Include paths from local build tree (ABI-matched with local libcuvs.so)
# ---------------------------------------------------------------------------
# CCCL (thrust, cub, libcudacxx) — must come before system CUDA headers
set(CCCL_SRC "${DEPS}/cccl-src")

# Mirror the exact include paths from the main cuVS build
# (extracted from compile_commands.json)
set(CONDA_PREFIX "/home/maxwb/miniconda3" CACHE PATH "Conda environment prefix")

set(BENCH_INCLUDE_DIRS
    # cuVS C++ and C API headers
    ${CUVS_ROOT}/include
    ${CUVS_ROOT}/../c/include
    # raft (source + generated)
    ${DEPS}/raft-src/cpp/include
    ${DEPS}/raft-build/include
    # rmm (source + generated)
    ${DEPS}/rmm-src/cpp/include
    ${DEPS}/rmm-build/include
    # CCCL (thrust, libcudacxx, cub)
    ${CCCL_SRC}/thrust
    ${CCCL_SRC}/libcudacxx/include
    ${CCCL_SRC}/cub
    # nvtx3
    ${DEPS}/nvtx3-src/c/include
    # conda system headers (rapids_logger, spdlog, dlpack, fmt)
    ${CONDA_PREFIX}/include
)

message(STATUS "BENCH_INCLUDE_DIRS: ${BENCH_INCLUDE_DIRS}")

# Common compile definitions (must match main cuVS build)
set(BENCH_COMPILE_DEFS
    CCCL_DISABLE_PDL
    CUB_DISABLE_NAMESPACE_MAGIC
    CUB_IGNORE_NAMESPACE_MAGIC_ERROR
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    THRUST_DISABLE_ABI_NAMESPACE
    THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
    THRUST_IGNORE_ABI_NAMESPACE_ERROR
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    RMM_DEPRECATE_LEGACY_MR_INTERFACE
    RMM_DEPRECATE_MR_DEVICE_HEADERS=1
    RMM_ENABLE_LEGACY_MR_INTERFACE
    RAFT_SYSTEM_LITTLE_ENDIAN=1
    RAFT_HIDE_DEPRECATION_WARNINGS
    CUDA_API_PER_THREAD_DEFAULT_STREAM
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
)

# Common CUDA compile options
set(BENCH_CUDA_FLAGS --extended-lambda --expt-relaxed-constexpr)

# Allow unresolved symbols at link time — libcuvs.so's transitive deps
# (librmm.so) are resolved at runtime via RPATH/LD_LIBRARY_PATH.
# This avoids ABI mismatches if rmm was rebuilt against a different CCCL.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# RPATH so the binary finds local libcuvs.so and librmm.so at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH
    ${CUVS_BUILD_DIR}
    ${DEPS}/rmm-build
    ${CU_ROARING_DIR}/build
)

# ---------------------------------------------------------------------------
# Helper function to reduce repetition
# ---------------------------------------------------------------------------
function(add_bench_target TARGET_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "ROARING" "" "" ${ARGN})
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE ${BENCH_INCLUDE_DIRS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>
    )
    if(ARG_ROARING)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CU_ROARING_DIR}/include
            ${CU_ROARING_DIR}/third_party/CRoaring/include)
        target_link_libraries(${TARGET_NAME} PRIVATE
            CUDA::cudart ${CUVS_LIB} ${RMM_LIB} ${CU_ROARING_LIB} ${CROARING_LIB}
        )
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE
            CUDA::cudart ${CUVS_LIB} ${RMM_LIB}
        )
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Benchmark targets
# ---------------------------------------------------------------------------

# CAGRA search: direct roaring_filter vs bitset_filter
add_bench_target(bench_cagra_roaring bench_cagra_roaring.cu ROARING)

# Comprehensive benchmark (selectivity sweep, multi-AND, E2E)
add_bench_target(bench_cagra_comprehensive bench_cagra_roaring_comprehensive.cu ROARING)

# Brute-force search: roaring_filter vs bitset_filter
add_bench_target(bench_brute_force_roaring bench_brute_force_roaring.cu ROARING)

# IVF-Flat search: roaring_filter vs bitset_filter
add_bench_target(bench_ivf_flat_roaring bench_ivf_flat_roaring.cu ROARING)

# Schedule-driven roaring search vs cuVS bitset filter.
# Uses the cu_roaring build that contains src/filtered_search.cu (the
# run-optimizations worktree); needs cuBLAS.
set(SD_ROARING_DIR "/tmp/cu-roaring-runopt" CACHE PATH "cu_roaring filtered_search source")
find_library(SD_ROARING_LIB cu_roaring_bitmap HINTS ${SD_ROARING_DIR}/build NO_DEFAULT_PATH)
find_library(SD_CROARING_LIB roaring
             HINTS ${SD_ROARING_DIR}/build/third_party/CRoaring/src NO_DEFAULT_PATH)
message(STATUS "SD_ROARING_LIB: ${SD_ROARING_LIB}")
if(SD_ROARING_LIB AND SD_CROARING_LIB)
    add_executable(bench_schedule_driven_roaring bench_schedule_driven_roaring.cu)
    target_include_directories(bench_schedule_driven_roaring PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${SD_ROARING_DIR}/include
        ${SD_ROARING_DIR}/third_party/CRoaring/include)
    target_compile_definitions(bench_schedule_driven_roaring PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(bench_schedule_driven_roaring PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(bench_schedule_driven_roaring PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>)
    target_link_libraries(bench_schedule_driven_roaring PRIVATE
        CUDA::cudart CUDA::cublas CUDA::curand ${CUVS_LIB} ${RMM_LIB}
        ${SD_ROARING_LIB} ${SD_CROARING_LIB})

    # Same harness, but loads filters from a directory of CRoaring portable
    # .bin files (synthetic generators produced by ../../../../roaring-benchmark).
    add_executable(bench_synth_filters bench_synth_filters.cu)
    target_include_directories(bench_synth_filters PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${SD_ROARING_DIR}/include
        ${SD_ROARING_DIR}/third_party/CRoaring/include)
    target_compile_definitions(bench_synth_filters PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(bench_synth_filters PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(bench_synth_filters PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>)
    target_link_libraries(bench_synth_filters PRIVATE
        CUDA::cudart CUDA::cublas CUDA::curand ${CUVS_LIB} ${RMM_LIB}
        ${SD_ROARING_LIB} ${SD_CROARING_LIB})

    # YFCC-10M real filtered search: per-query schedule build + search.
    add_executable(bench_yfcc_search bench_yfcc_search.cu)
    target_include_directories(bench_yfcc_search PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${SD_ROARING_DIR}/include
        ${SD_ROARING_DIR}/third_party/CRoaring/include)
    target_compile_definitions(bench_yfcc_search PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(bench_yfcc_search PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(bench_yfcc_search PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>)
    target_link_libraries(bench_yfcc_search PRIVATE
        CUDA::cudart CUDA::cublas ${CUVS_LIB} ${RMM_LIB}
        ${SD_ROARING_LIB} ${SD_CROARING_LIB})

    # End-to-end selectivity sweep: cuVS bitset vs cu_roaring schedule-driven
    # search, with H2D copy + setup + search + D2H all inside the timed region.
    add_executable(bench_e2e_sweep bench_e2e_sweep.cu)
    target_include_directories(bench_e2e_sweep PRIVATE
        ${BENCH_INCLUDE_DIRS}
        ${SD_ROARING_DIR}/include
        ${SD_ROARING_DIR}/third_party/CRoaring/include)
    target_compile_definitions(bench_e2e_sweep PRIVATE ${BENCH_COMPILE_DEFS})
    set_target_properties(bench_e2e_sweep PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(bench_e2e_sweep PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>)
    target_link_libraries(bench_e2e_sweep PRIVATE
        CUDA::cudart CUDA::cublas CUDA::curand ${CUVS_LIB} ${RMM_LIB}
        ${SD_ROARING_LIB} ${SD_CROARING_LIB})
else()
    message(WARNING "cu_roaring_bitmap lib not found — skipping bench_schedule_driven_roaring")
endif()

# CAGRA profiling harness for nsight compute
add_bench_target(profile_cagra_roaring profile_cagra_roaring.cu ROARING)

# Transfer + search latency comparison (bitset upload vs CRoaring upload)
add_bench_target(bench_transfer bench_transfer.cu ROARING)
target_link_libraries(profile_cagra_roaring PRIVATE CUDA::cudart_static)

# Standalone roaring primitives benchmark (upload, decompress, etc.)
add_executable(roaring_bench
    roaring_bench.cu
    ../../../src/core/roaring/roaring.cu
)
target_include_directories(roaring_bench PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(roaring_bench PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB})
target_compile_definitions(roaring_bench PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(roaring_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(roaring_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${BENCH_CUDA_FLAGS}>
)
