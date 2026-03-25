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

# Use conda-installed headers + libs for ABI consistency.
# Local cuVS include dir for roaring_filter.cuh and roaring.hpp (not in conda).
set(CONDA_PREFIX "/home/maxwb/miniconda3" CACHE PATH "Conda environment prefix")
set(CUVS_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../build" CACHE PATH "cuVS build directory")

# All headers from conda (consistent ABI with conda libcuvs + librmm)
set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
# Local cuVS headers for roaring integration (header-only, no ABI dependency)
set(CUVS_LOCAL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")

message(STATUS "CONDA_INCLUDE: ${CONDA_INCLUDE}")
message(STATUS "CUVS_LOCAL_INCLUDE: ${CUVS_LOCAL_INCLUDE}")

# Libraries from conda
find_library(RMM_LIB rmm HINTS ${CONDA_PREFIX}/lib)
find_library(CUVS_LIB cuvs HINTS ${CONDA_PREFIX}/lib)
message(STATUS "RMM_LIB: ${RMM_LIB}")
message(STATUS "CUVS_LIB: ${CUVS_LIB}")

# CCCL from conda (in rapids/ subdir) — must go first to override system CUDA's CCCL
set(CONDA_RAPIDS "${CONDA_PREFIX}/include/rapids")

# Common include dirs: conda rapids CCCL first (overrides system CUDA CCCL),
# then local cuVS for roaring headers, then conda standard headers.
set(BENCH_INCLUDE_DIRS
    ${CONDA_RAPIDS}
    ${CUVS_LOCAL_INCLUDE}
    ${CONDA_INCLUDE}
)

# Common compile options
set(BENCH_COMPILE_DEFS
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)

# Build benchmark + roaring source
add_executable(roaring_bench
    roaring_bench.cu
    ../../../src/core/roaring/roaring.cu
)
target_include_directories(roaring_bench PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(roaring_bench PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB})
target_compile_definitions(roaring_bench PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(roaring_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(roaring_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# cu-roaring-filter standalone library
set(CU_ROARING_DIR "/mnt/c/Users/maxwb/Development/cu-roaring-filter" CACHE PATH "cu-roaring-filter dir")
find_library(CU_ROARING_LIB cu_roaring_filter HINTS ${CU_ROARING_DIR}/build)
find_library(CROARING_LIB roaring HINTS ${CU_ROARING_DIR}/build/third_party/CRoaring/src)
message(STATUS "CU_ROARING_LIB: ${CU_ROARING_LIB}")
message(STATUS "CROARING_LIB: ${CROARING_LIB}")

# CAGRA + Brute Force search benchmark with roaring_filter
add_executable(bench_cagra_roaring bench_cagra_roaring.cu)
target_include_directories(bench_cagra_roaring PRIVATE ${BENCH_INCLUDE_DIRS} ${CU_ROARING_DIR}/include)
target_link_libraries(bench_cagra_roaring PRIVATE
    CUDA::cudart ${RMM_LIB} ${CUVS_LIB} ${CU_ROARING_LIB} ${CROARING_LIB}
)
target_compile_definitions(bench_cagra_roaring PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(bench_cagra_roaring PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_cagra_roaring PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# Comprehensive CAGRA + Roaring benchmark (selectivity sweep, multi-AND, E2E)
add_executable(bench_cagra_comprehensive bench_cagra_roaring_comprehensive.cu)
target_include_directories(bench_cagra_comprehensive PRIVATE ${BENCH_INCLUDE_DIRS} ${CU_ROARING_DIR}/include)
target_link_libraries(bench_cagra_comprehensive PRIVATE
    CUDA::cudart ${RMM_LIB} ${CUVS_LIB} ${CU_ROARING_LIB} ${CROARING_LIB}
)
target_compile_definitions(bench_cagra_comprehensive PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(bench_cagra_comprehensive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_cagra_comprehensive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# Minimal CAGRA test
add_executable(quick_test quick_test.cu)
target_include_directories(quick_test PRIVATE ${BENCH_INCLUDE_DIRS})
target_link_libraries(quick_test PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB})
target_compile_definitions(quick_test PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(quick_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(quick_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# Minimal crash test
add_executable(test_upload_crash test_upload_crash.cu)
target_include_directories(test_upload_crash PRIVATE ${BENCH_INCLUDE_DIRS} ${CU_ROARING_DIR}/include)
target_link_libraries(test_upload_crash PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB} ${CU_ROARING_LIB} ${CROARING_LIB})
target_compile_definitions(test_upload_crash PRIVATE ${BENCH_COMPILE_DEFS})
set_target_properties(test_upload_crash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_upload_crash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
