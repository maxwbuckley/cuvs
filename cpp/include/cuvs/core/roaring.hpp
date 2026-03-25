/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/bitset.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <memory>

namespace cuvs::core {

/**
 * @defgroup roaring GPU Roaring Bitmap
 *
 * GPU-native Roaring Bitmap for compressed prefiltering in vector search.
 * Stores attribute bitmaps in compressed Roaring format in GPU memory,
 * supports bulk set operations (AND/OR/ANDNOT/XOR), and decompresses
 * to flat bitsets compatible with cuvs::core::bitset for search.
 *
 * Typical usage:
 * @code{.cpp}
 *   // Build filter bitmaps on CPU, upload to GPU
 *   auto category = gpu_roaring::from_sorted_ids(res, cat_ids, universe);
 *   auto price    = gpu_roaring::from_sorted_ids(res, price_ids, universe);
 *
 *   // Combine on GPU
 *   auto combined = gpu_roaring::set_and(res, category, price);
 *
 *   // Decompress to bitset for search
 *   auto bitset = gpu_roaring::to_bitset(res, combined);
 *   auto filter = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitset.view());
 *   cuvs::neighbors::cagra::search(res, params, index, queries, neighbors, distances, filter);
 * @endcode
 * @{
 */

/** Container type tags for Roaring containers */
enum class roaring_container_type : uint8_t {
  ARRAY  = 0,
  BITMAP = 1,
  RUN    = 2
};

/**
 * @brief GPU-resident Roaring bitmap in Structure-of-Arrays layout.
 *
 * All device memory is managed via rmm::device_uvector (RAII).
 * Supports move semantics; not copyable.
 */
struct gpu_roaring {
  // Top-level index (one entry per container, sorted by key)
  rmm::device_uvector<uint16_t> keys;
  rmm::device_uvector<roaring_container_type> types;
  rmm::device_uvector<uint32_t> offsets;
  rmm::device_uvector<uint16_t> cardinalities;
  uint32_t n_containers = 0;
  uint32_t universe_size = 0;

  // Per-type data pools
  rmm::device_uvector<uint64_t> bitmap_data;
  uint32_t n_bitmap_containers = 0;

  rmm::device_uvector<uint16_t> array_data;
  uint32_t n_array_containers = 0;

  rmm::device_uvector<uint16_t> run_data;
  uint32_t n_run_containers = 0;

  // Direct-map key index: key_index[high16] = container index, or 0xFFFF.
  // Replaces O(log n) binary search with O(1) table lookup.
  rmm::device_uvector<uint16_t> key_index;
  uint32_t max_key = 0;

  // Complement optimization: when true, the stored set is the complement
  // of the logical set. contains() results are flipped at query time.
  bool negated = false;

  // Total logical cardinality (number of set bits, before complement)
  uint64_t total_cardinality = 0;

  /** Construct an empty GPU Roaring bitmap */
  explicit gpu_roaring(rmm::cuda_stream_view stream)
    : keys(0, stream),
      types(0, stream),
      offsets(0, stream),
      cardinalities(0, stream),
      bitmap_data(0, stream),
      array_data(0, stream),
      run_data(0, stream),
      key_index(0, stream) {}

  gpu_roaring(gpu_roaring&&) = default;
  gpu_roaring& operator=(gpu_roaring&&) = default;

  /** Total device memory used (bytes) */
  [[nodiscard]] size_t device_memory_bytes() const
  {
    return keys.size() * sizeof(uint16_t) + types.size() * sizeof(roaring_container_type) +
           offsets.size() * sizeof(uint32_t) + cardinalities.size() * sizeof(uint16_t) +
           bitmap_data.size() * sizeof(uint64_t) + array_data.size() * sizeof(uint16_t) +
           run_data.size() * sizeof(uint16_t) + key_index.size() * sizeof(uint16_t);
  }

  /** Equivalent flat bitset size (bytes) */
  [[nodiscard]] size_t flat_bitset_bytes() const
  {
    return (static_cast<size_t>(universe_size) + 31) / 32 * sizeof(uint32_t);
  }

  /** Compression ratio (flat / compressed) */
  [[nodiscard]] double compression_ratio() const
  {
    auto dev_bytes = device_memory_bytes();
    return dev_bytes > 0 ? static_cast<double>(flat_bitset_bytes()) / dev_bytes : 0.0;
  }
};

/** Set operation types */
enum class roaring_set_op : uint8_t {
  AND    = 0,
  OR     = 1,
  ANDNOT = 2,
  XOR    = 3
};

/**
 * @brief Create a GPU Roaring bitmap from a sorted array of IDs on host.
 *
 * The IDs are partitioned into 65536-element containers following the
 * Roaring bitmap format. Containers with <= 4096 elements use array format;
 * denser containers use bitmap format.
 *
 * @param[in] res RAFT resources (provides CUDA stream)
 * @param[in] sorted_ids Host pointer to sorted, deduplicated uint32_t IDs
 * @param[in] n_ids Number of IDs
 * @param[in] universe_size Maximum representable ID + 1
 * @return GPU Roaring bitmap
 */
gpu_roaring from_sorted_ids(raft::resources const& res,
                            const uint32_t* sorted_ids,
                            uint32_t n_ids,
                            uint32_t universe_size);

/**
 * @brief Perform a pairwise set operation between two GPU Roaring bitmaps.
 *
 * @param[in] res RAFT resources
 * @param[in] a First operand
 * @param[in] b Second operand
 * @param[in] op Set operation (AND, OR, ANDNOT, XOR)
 * @return Result GPU Roaring bitmap
 */
gpu_roaring set_op(raft::resources const& res,
                   const gpu_roaring& a,
                   const gpu_roaring& b,
                   roaring_set_op op);

/** Convenience: result = a AND b */
inline gpu_roaring set_and(raft::resources const& res,
                           const gpu_roaring& a,
                           const gpu_roaring& b)
{
  return set_op(res, a, b, roaring_set_op::AND);
}

/** Convenience: result = a OR b */
inline gpu_roaring set_or(raft::resources const& res,
                          const gpu_roaring& a,
                          const gpu_roaring& b)
{
  return set_op(res, a, b, roaring_set_op::OR);
}

/**
 * @brief Multi-bitmap AND: result = a[0] AND a[1] AND ... AND a[n-1].
 */
gpu_roaring multi_and(raft::resources const& res,
                      const gpu_roaring* bitmaps,
                      uint32_t count);

/**
 * @brief Multi-bitmap OR: result = a[0] OR a[1] OR ... OR a[n-1].
 */
gpu_roaring multi_or(raft::resources const& res,
                     const gpu_roaring* bitmaps,
                     uint32_t count);

/**
 * @brief Decompress a GPU Roaring bitmap to a flat bitset.
 *
 * The output is compatible with cuvs::core::bitset<uint32_t, int64_t>
 * and can be wrapped in a bitset_filter for use with cuVS search.
 *
 * @param[in] res RAFT resources
 * @param[in] bitmap Source GPU Roaring bitmap
 * @return Flat bitset (device memory, RAII)
 */
cuvs::core::bitset<uint32_t, int64_t> to_bitset(raft::resources const& res,
                                                 const gpu_roaring& bitmap);

/**
 * @brief Decompress into a pre-allocated flat bitset buffer.
 *
 * @param[in]  res RAFT resources
 * @param[in]  bitmap Source GPU Roaring bitmap
 * @param[out] output Device pointer to uint32_t array, pre-zeroed
 * @param[in]  output_size_words Number of uint32_t words in output
 */
void decompress_to_bitset(raft::resources const& res,
                          const gpu_roaring& bitmap,
                          uint32_t* output,
                          uint32_t output_size_words);

/** @} */  // end group roaring

}  // namespace cuvs::core
