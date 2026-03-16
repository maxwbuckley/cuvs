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

# Find raft and rmm via the cuVS build
set(CUVS_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../build" CACHE PATH "cuVS build directory")

# raft headers
find_path(RAFT_INCLUDE_DIR raft/core/resources.hpp
    HINTS
    ${CUVS_BUILD_DIR}/_deps/raft-src/cpp/include
    /home/maxwb/miniconda3/include
)

# rmm headers
find_path(RMM_INCLUDE_DIR rmm/device_uvector.hpp
    HINTS
    ${CUVS_BUILD_DIR}/_deps/rmm-src/cpp/include
    ${CUVS_BUILD_DIR}/_deps/rmm-src/include
    /home/maxwb/miniconda3/include
)

# cuvs headers
set(CUVS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")

# CCCL/thrust/libcudacxx headers (from build)
find_path(CCCL_INCLUDE_DIR thrust/device_vector.h
    HINTS
    ${CUVS_BUILD_DIR}/_deps/cccl-src/thrust
    ${CUVS_BUILD_DIR}/_deps/cccl-src/thrust/thrust
    /home/maxwb/miniconda3/include
)
find_path(LIBCUDACXX_INCLUDE_DIR cuda/std/type_traits
    HINTS
    ${CUVS_BUILD_DIR}/_deps/cccl-src/libcudacxx/include
    /home/maxwb/miniconda3/include
)
find_path(CUB_INCLUDE_DIR cub/cub.cuh
    HINTS
    ${CUVS_BUILD_DIR}/_deps/cccl-src/cub
    /home/maxwb/miniconda3/include
)

message(STATUS "RAFT_INCLUDE_DIR: ${RAFT_INCLUDE_DIR}")
message(STATUS "RMM_INCLUDE_DIR: ${RMM_INCLUDE_DIR}")
message(STATUS "CUVS_INCLUDE_DIR: ${CUVS_INCLUDE_DIR}")
message(STATUS "CCCL thrust: ${CCCL_INCLUDE_DIR}")
message(STATUS "libcudacxx: ${LIBCUDACXX_INCLUDE_DIR}")
message(STATUS "CUB: ${CUB_INCLUDE_DIR}")

# Build benchmark + roaring source
add_executable(roaring_bench
    roaring_bench.cu
    ../../../src/core/roaring/roaring.cu
)

target_include_directories(roaring_bench PRIVATE
    ${CUVS_INCLUDE_DIR}
    ${RAFT_INCLUDE_DIR}
    ${RMM_INCLUDE_DIR}
    ${CCCL_INCLUDE_DIR}
    ${LIBCUDACXX_INCLUDE_DIR}
    ${CUB_INCLUDE_DIR}
)

# Link against rmm from the cuVS build
find_library(RMM_LIB rmm HINTS ${CUVS_BUILD_DIR}/_deps/rmm-build)
message(STATUS "RMM_LIB: ${RMM_LIB}")

# Link against libcuvs for raft::core::bitset template instantiations
find_library(CUVS_LIB cuvs HINTS ${CUVS_BUILD_DIR})
message(STATUS "CUVS_LIB: ${CUVS_LIB}")

target_link_libraries(roaring_bench PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB})

# RMM uses spdlog
find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h
    HINTS
    ${CUVS_BUILD_DIR}/_deps/spdlog-src/include
    /home/maxwb/miniconda3/include
)
find_path(FMT_INCLUDE_DIR fmt/format.h
    HINTS
    ${CUVS_BUILD_DIR}/_deps/fmt-src/include
    /home/maxwb/miniconda3/include
)
if(SPDLOG_INCLUDE_DIR)
    target_include_directories(roaring_bench PRIVATE ${SPDLOG_INCLUDE_DIR})
endif()
if(FMT_INCLUDE_DIR)
    target_include_directories(roaring_bench PRIVATE ${FMT_INCLUDE_DIR})
endif()

target_compile_definitions(roaring_bench PRIVATE
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
)

set_target_properties(roaring_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(roaring_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# cu-roaring-filter standalone library
set(CU_ROARING_DIR "/mnt/c/Users/maxwb/Development/cu-roaring-filter" CACHE PATH "cu-roaring-filter dir")
find_library(CU_ROARING_LIB cu_roaring_filter HINTS ${CU_ROARING_DIR}/build)
message(STATUS "CU_ROARING_LIB: ${CU_ROARING_LIB}")

# CAGRA + Brute Force search benchmark with roaring_filter
add_executable(bench_cagra_roaring bench_cagra_roaring.cu)
target_include_directories(bench_cagra_roaring PRIVATE
    ${CUVS_INCLUDE_DIR}
    ${RAFT_INCLUDE_DIR}
    ${RMM_INCLUDE_DIR}
    ${CCCL_INCLUDE_DIR}
    ${LIBCUDACXX_INCLUDE_DIR}
    ${CUB_INCLUDE_DIR}
    ${CU_ROARING_DIR}/include
)
target_link_libraries(bench_cagra_roaring PRIVATE
    CUDA::cudart ${RMM_LIB} ${CUVS_LIB} ${CU_ROARING_LIB}
)
if(SPDLOG_INCLUDE_DIR)
    target_include_directories(bench_cagra_roaring PRIVATE ${SPDLOG_INCLUDE_DIR})
endif()
if(FMT_INCLUDE_DIR)
    target_include_directories(bench_cagra_roaring PRIVATE ${FMT_INCLUDE_DIR})
endif()
target_compile_definitions(bench_cagra_roaring PRIVATE
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)
set_target_properties(bench_cagra_roaring PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bench_cagra_roaring PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

# Minimal CAGRA test
add_executable(quick_test quick_test.cu)
target_include_directories(quick_test PRIVATE
    ${CUVS_INCLUDE_DIR}
    ${RAFT_INCLUDE_DIR}
    ${RMM_INCLUDE_DIR}
    ${CCCL_INCLUDE_DIR}
    ${LIBCUDACXX_INCLUDE_DIR}
    ${CUB_INCLUDE_DIR}
)
target_link_libraries(quick_test PRIVATE CUDA::cudart ${RMM_LIB} ${CUVS_LIB})
if(SPDLOG_INCLUDE_DIR)
    target_include_directories(quick_test PRIVATE ${SPDLOG_INCLUDE_DIR})
endif()
if(FMT_INCLUDE_DIR)
    target_include_directories(quick_test PRIVATE ${FMT_INCLUDE_DIR})
endif()
target_compile_definitions(quick_test PRIVATE
    FMT_HEADER_ONLY=1
    SPDLOG_FMT_EXTERNAL=0
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)
set_target_properties(quick_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(quick_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
