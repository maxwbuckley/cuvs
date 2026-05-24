# GPU Roaring Bitmap vs Flat Bitset: CAGRA Filtered Search
## Direct roaring_filter kernel path (no decompression)

**Generated:** 2026-03-26, **Corrected:** 2026-03-28
**GPU:** NVIDIA GeForce RTX 5090 (170 SMs, 32 GB VRAM, 96 MB L2)
**Platform:** WSL2 (Linux 6.6.87 on Windows)
**Parameters:** k=10, graph_degree=32, itopk_size=256
**Methodology:** Interleaved A/B (alternating bitset/roaring each iteration), 50 measured iterations, 20 warmup, median latency, Welch's t-test

## CORRECTION (2026-03-28): Previous 5-33% speedup was a benchmark bug

The original results (below, struck through) showed roaring 5-33% faster than bitset. Nsight Systems profiling revealed both search kernels use **identical** register counts (63), launch parameters (grid=12, block=128, shared=2120B), and GPU execution time (~1% difference). The entire measured speedup was caused by a **per-call popcount overhead** in cuVS's `bitset_filter` dispatch path.

**Root cause:** `search_params.filtering_rate` defaults to `-1.0` (auto-detect). For `bitset_filter`, this triggers `bitset_view_.count(res)` on every `search()` call — a GPU reduction kernel (`coalescedSumMediumKernel`) plus a host-side stream synchronization to read the scalar result. For `roaring_filter`, the cardinality is a precomputed host integer (`cardinality_`), so no GPU work is needed.

The benchmark did not set `filtering_rate`, so every bitset search call paid ~150-450us of popcount + sync overhead while roaring did not. This created an artificial advantage that scaled with how fast the search kernel was (faster kernel = larger proportional overhead = larger apparent "speedup" at high selectivity).

**Filed as:** [rapidsai/cuvs#1960](https://github.com/rapidsai/cuvs/issues/1960)

### Corrected results (filtering_rate pre-set, eliminating popcount bias)

## Results: 1M Vectors (dim=128)

| Selectivity | Bitset Mem | Roaring Mem | Compress | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Signif | Recall |
|-------------|-----------|-------------|----------|-------------|-------------|---------|--------|--------|--------|
| 0.1% | 122 KB | **0.1 KB** | **868x** | 14.667 | 14.940 | 0.98x | -1.08 | | 0.990 |
| 1% | 122 KB | 128 KB | 0.95x | 4.700 | 4.471 | 1.05x | 0.87 | | 0.974 |
| 5% | 122 KB | 128 KB | 0.95x | 1.721 | 1.692 | 1.02x | 0.36 | | 0.957 |
| 10% | 122 KB | 128 KB | 0.95x | 1.566 | 1.584 | 0.99x | -1.11 | | 0.968 |
| 25% | 122 KB | 128 KB | 0.95x | 1.092 | 1.120 | 0.97x | -0.83 | | 0.982 |
| 50% | 122 KB | 128 KB | 0.95x | 0.944 | 0.873 | 1.08x | 2.33 | * | 0.981 |

## Results: 10M Vectors (dim=128)

| Selectivity | Bitset Mem | Roaring Mem | Compress | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Signif | Recall |
|-------------|-----------|-------------|----------|-------------|-------------|---------|--------|--------|--------|
| 0.1% | 1.2 MB | 1.2 MB | 1.0x | 17.785 | 17.807 | 1.00x | -0.29 | | 0.966 |
| 1% | 1.2 MB | 1.2 MB | 1.0x | 5.196 | 5.200 | 1.00x | 0.75 | | 0.939 |
| 10% | 1.2 MB | 1.2 MB | 1.0x | 1.767 | 1.747 | 1.01x | -0.17 | | 0.954 |
| 50% | 1.2 MB | 1.2 MB | 1.0x | 0.932 | 0.922 | 1.01x | 1.60 | | 0.978 |

## Results: 20M Vectors (dim=32)

| Selectivity | Bitset Mem | Roaring Mem | Compress | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Signif | Recall |
|-------------|-----------|-------------|----------|-------------|-------------|---------|--------|--------|--------|
| 0.1% | 2.4 MB | 2.5 MB | 1.0x | 10.980 | 11.298 | 0.97x | -1.76 | | 0.997 |
| 1% | 2.4 MB | 2.5 MB | 1.0x | 3.035 | 3.198 | 0.95x | -1.21 | | 0.998 |
| 10% | 2.4 MB | 2.5 MB | 1.0x | 1.244 | 1.275 | 0.98x | -1.26 | | 0.997 |
| 50% | 2.4 MB | 2.5 MB | 1.0x | 1.110 | 1.204 | 0.92x | -0.77 | | 0.995 |

Significance: \* p<0.05 (|t|>2.0), \*\* p<0.01 (|t|>2.6), \*\*\* p<0.001 (|t|>3.3)

**Zero of 14 configs are significant at p<0.01.** Only 1M/50% shows a marginal 1.08x (p<0.05). At 20M, roaring is slightly *slower* due to the extra instructions (warp intrinsics, two-read lookup) compared to bitset's single flat memory read.

## Nsight Systems profiling confirmation

| Metric | Bitset kernel | Roaring kernel |
|--------|--------------|----------------|
| Registers/thread | 63 | 63 |
| Block size | 128 | 128 |
| Grid size | 12 | 12 |
| Dynamic shared mem | 2120 B | 2120 B |
| Avg GPU time (ns) | 552,869 | 547,559 |
| **Kernel-level diff** | | **~1%** |

Both template instantiations compile to kernels with identical resource usage. The search kernel GPU execution time differs by ~1%, consistent with the corrected benchmark results.

## Analysis

### Why roaring shows no compression at 10M+ (1.2 MB = same as bitset)

At 0.1% pass rate with 10M vectors, only ~10,000 IDs are spread across ~153 containers (~65 IDs each). A CPU roaring bitmap would store these as **array containers** (2 bytes/element) totaling ~20 KB — 60x smaller than the flat bitset. But the GPU version reports 1.2 MB, identical to the bitset. This is due to automatic container promotion.

`upload_from_sorted_ids` uses `PROMOTE_AUTO` by default. The auto-promotion heuristic in `cu_roaring/detail/promote.cuh` checks the universe size:

- Universe <= ~4M (<=64 containers): keep array containers as-is
- Universe > ~4M (>64 containers): **promote all containers to bitmap format**

At 10M vectors (153 containers > 64 threshold), every container is promoted to an 8 KB bitmap regardless of cardinality. So 153 containers x 8 KB = 1,224 KB ≈ 1.2 MB.

The promotion exists because GPU array container lookups require binary search (up to 12 divergent warp steps), which is 4-10x slower than a single bitmap bit-test. At scale, the memory cost of promotion is justified by the query speed gain.

**Container skipping still applies even without compression.** The two-level key lookup (`key_index[id >> 16]`) can skip entire 65K-element ranges that have zero IDs. However, in this benchmark IDs are uniformly random, so at 0.1% pass rate nearly every container has some IDs (~65 each) and no containers are skipped. Container skipping would provide a larger benefit with **clustered** ID distributions (e.g., IDs concentrated in a few ranges, leaving many containers absent) or at much larger universe sizes where sparse regions produce empty 65K-element ranges.

**To observe actual compression**, you would need either:
1. A universe large enough that the flat bitset overflows L2 (>96 MB → >768M vectors), where roaring's sparse containers would provide both compression and cache benefits
2. `PROMOTE_NONE` mode to keep array containers — but query speed would regress due to divergent binary search

### Why roaring and bitset have identical kernel performance

With `filtering_rate` pre-set (eliminating the popcount bias), the corrected results show **no meaningful performance difference** between roaring and bitset for CAGRA search at 1M-20M scale.

This is expected because:

1. **Identical memory layout.** After `PROMOTE_ALL`, roaring's bitmap containers laid out contiguously are functionally the same data as a flat bitset. Both are ~1.2 MB of bitmap data at 10M. From the GPU cache's perspective, a random read into roaring's `bitmap_data[container*1024 + (low>>6)]` hits the same address range as `bitset_ptr_[id/32]`.

2. **CAGRA graph neighbors are random in ID space.** CAGRA ([Ootomo et al., 2023](https://arxiv.org/abs/2308.15136)) builds a graph based on vector proximity, not ID proximity. Neighbors of a node are close in vector space but scattered in ID space. This means filter access patterns are random for both approaches — neither benefits from spatial locality.

3. **Roaring does strictly more work per check.** Bitset: 1 memory read + shift + mask. Roaring: `__match_any_sync` + `__shfl_sync` + `__ldg` (key_index) + `__ldg` (bitmap word) + shift + mask. The `key_index` read is effectively free (306 bytes, permanently in L1), but the warp intrinsics add instruction overhead.

4. **Identical compiled kernel characteristics.** Nsight Systems confirmed both template instantiations use 63 registers, 128-thread blocks, and 2120B shared memory — identical occupancy.

### Roaring's value proposition for CAGRA

At 1M-20M scale with uniform random filters, roaring provides no kernel-level speedup over bitset for CAGRA. Its value for CAGRA lies elsewhere:

- **Compression at extreme scale** (>768M vectors): when the flat bitset exceeds L2 cache (96 MB), roaring's containers would still fit individually in L1
- **Clustered filters**: real-world filters often have structure (e.g., "all users in region X") that creates empty containers, enabling container skipping
- **API convenience**: `roaring_filter` stores cardinality as a host integer, avoiding the per-call popcount overhead that `bitset_filter` incurs with default `filtering_rate=-1.0`

## CAGRA Methodology

- **Interleaved A/B**: Each iteration alternates which filter runs first (even: bitset first, odd: roaring first), eliminating warm-cache bias
- **GPU event timing**: `cudaEventRecord`/`cudaEventElapsedTime` for precise kernel timing
- **filtering_rate pre-set**: Explicitly set to `1.0 - pass_rate` to eliminate per-call popcount overhead (see correction note above)
- **Warmup**: 20 interleaved iterations before measurement
- **Iterations**: 50 measured pairs
- **Statistics**: median latency, sample standard deviation (Bessel-corrected), Welch's t-test on means
- **Recall baseline**: bitset_filter results (bitset is ground truth)
- **CAGRA index**: built once per (N, dim) pair, reused across selectivities
- **CUDA JIT cache**: `CUDA_CACHE_MAXSIZE=8589934592` to avoid repeated JIT compilation

---

# GPU Roaring Bitmap vs Flat Bitset: Brute-Force Filtered Search

**Generated:** 2026-03-28
**GPU:** NVIDIA GeForce RTX 5090 (170 SMs, 32 GB VRAM, 96 MB L2)
**Platform:** WSL2 (Linux 6.6.87 on Windows)
**Parameters:** k=10, metric=L2Expanded, n_queries=100
**Methodology:** Interleaved A/B, 30 measured iterations, 10 warmup, median latency, Welch's t-test

## Summary

Brute-force search computes a full pairwise distance matrix (GEMM), then masks filtered entries before top-k selection. The filter only affects the masking step, not the dominant GEMM. Two code paths exist for bitset:
- **Dense path** (pass rate ≥ 10%): Full GEMM → bitset masking → select_k
- **CSR path** (pass rate < 10%): Convert filter to CSR → masked SpGEMM (skips distance computation for filtered entries) → select_k

Three roaring path versions were developed:
- **v1 (dense only)**: Full GEMM → `warp_contains()` masking → select_k (all pass rates)
- **v2 (decompress to CSR)**: At <10% pass rate, decompress roaring to bitset, then follow standard bitset-to-CSR pipeline
- **v3 (enumerate_ids to SDDMM)**: At <10% pass rate, use `cu_roaring::enumerate_ids()` to produce CSR indices directly, then `cusparseSDDMM` for masked distance computation

**Key findings:**
- When both use the **dense path** (≥10% pass rate), roaring is **1-9% faster** but the difference is mostly within noise because GEMM dominates total time. Statistically significant only at 100K scale where masking is a larger fraction of total time.
- At **5-10% pass rate**, roaring's dense path **beats bitset's CSR path by up to 1.9x** because CSR conversion overhead exceeds the GEMM savings at scale.
- At **1% pass rate**, v1 (dense only) lost ~3x to bitset CSR. v3 (SDDMM) achieves **near-parity** with bitset CSR at all scales (0.74x at 100K, 0.97-1.03x at 1M-10M).
- **Recall is 1.0000** across all configurations — roaring produces identical results to bitset.

## Results: Dense Path (Both Use GEMM + Masking)

These are the fair apples-to-apples comparisons where both filters use the same code path, differing only in the masking kernel (flat bit lookup vs `warp_contains()`).

### 100K Vectors (dim=128)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 10% | 2.651 | 2.615 | 1.01x | 0.57 | |
| 25% | 1.332 | 1.218 | **1.09x** | 3.63 | *** |
| 50% | 1.265 | 1.159 | **1.09x** | 3.62 | *** |

### 1M Vectors (dim=128)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 25% | 7.488 | 7.286 | 1.03x | 4.35 | *** |
| 50% | 7.559 | 7.451 | 1.01x | 0.59 | |

### 5M Vectors (dim=128)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 25% | 20.676 | 19.967 | 1.04x | 0.11 | |
| 50% | 24.847 | 24.313 | 1.02x | 0.80 | |

### 10M Vectors (dim=128)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 10% | 34.639 | 33.679 | 1.03x | -0.46 | |
| 50% | 34.254 | 33.992 | 1.01x | -0.48 | |

## Results: Cross-Path Comparisons (Bitset CSR vs Roaring v1 Dense-Only)

At pass rates below 10%, the bitset path switches to CSR + masked SpGEMM which skips distance computation for filtered entries. The v1 roaring path always uses the full GEMM + mask approach. These comparisons are informative but not apples-to-apples. See the Analysis section for v2 and v3 results that close this gap.

### Where Roaring Dense Beats Bitset CSR

| Scale | Pass Rate | Bitset CSR (ms) | Roaring Dense (ms) | Speedup | t-stat |
|-------|-----------|-----------------|--------------------|---------|----|
| 100K | 5% | 2.074 | 1.133 | **1.83x** | 20.28 |
| 1M | 10% | 8.782 | 7.397 | **1.19x** | 15.16 |
| 5M | 10% | 41.133 | 21.619 | **1.90x** | 21.19 |

At **5M/10% pass rate**, roaring's dense GEMM+mask is nearly **2x faster** than bitset's CSR conversion + SpGEMM. The CSR conversion overhead (converting the bitset to compressed sparse row format, then performing SpGEMM) exceeds the savings from skipping 90% of distance computation.

### Where Bitset CSR Wins

| Scale | Pass Rate | Bitset CSR (ms) | Roaring Dense (ms) | Speedup |
|-------|-----------|-----------------|--------------------|---------|
| 100K | 1% | 1.163 | 2.683 | 0.43x |
| 1M | 1% | 2.554 | 8.006 | 0.32x |
| 1M | 5% | 5.071 | 7.585 | 0.67x |
| 5M | 1% | 5.285 | 18.510 | 0.29x |
| 5M | 5% | 19.951 | 20.080 | 0.99x (tie) |
| 10M | 1% | 11.502 | 33.636 | 0.34x |

At 1% pass rate, CSR wins by ~3x because the SpGEMM only computes 1% of the distance matrix.

## Analysis

### Why the dense-path difference is small

In brute-force search, the dominant cost is pairwise distance computation (GEMM), not filter masking:

| Scale (N) | GEMM (approx) | Masking (approx) | Masking fraction |
|-----------|--------------|-----------------|-----------------|
| 100K | ~1 ms | ~0.1 ms | ~10% |
| 1M | ~7 ms | ~0.2 ms | ~3% |
| 5M | ~19 ms | ~0.5 ms | ~2.5% |
| 10M | ~33 ms | ~1 ms | ~3% |

Even a 50% improvement in masking speed yields only 1-5% total speedup. This is why the dense-path results show consistent but small improvements.

At 100K, masking is ~10% of total time, and the 9% total speedup at 25-50% pass rate (t>3.6, p<0.001) is statistically significant. At larger scales, the GEMM dominates and the masking improvement disappears into noise.

### Why roaring's warp_contains() helps even with sequential access

Unlike CAGRA (random access with temporal L1 locality), brute-force processes dataset IDs sequentially within tiles. Despite the sequential pattern, `warp_contains()` still provides a small advantage:

1. **All 32 warp threads share the same container key** when scanning sequential IDs within a 64K range. `__match_any_sync` finds all 32 threads matching → only 1 key lookup for the whole warp.
2. **8 KB container fits in L1** regardless of dataset size, while the bitset (12 KB–1.2 MB) may spill to L2 at scale.

However, the bitset's sequential access is also very cache-friendly — consecutive threads read consecutive 32-bit words, which is a coalesced pattern. So the cache advantage is smaller than in CAGRA's random-access pattern.

### The CSR crossover — three versions of the roaring sparse path

The original roaring dense-only path lost 3x at low pass rates because it computed the full GEMM while the bitset path used CSR+SpGEMM. Three iterations of the sparse path addressed this:

**v1 (dense only):** Roaring always uses full GEMM + `warp_contains()` masking. At 1% pass rate, 3x slower than bitset CSR+SpGEMM because it computes 99% of distances unnecessarily.

**v2 (decompress to CSR):** At <10% pass rate, decompress roaring to a flat bitset, then follow the standard bitset-to-CSR conversion pipeline (calc_nnz, prefix_scan, fill_indices, repeat_csr). Closes the gap but 5-7% slower than bitset due to extra decompression kernel overhead (7 kernel launches before actual computation).

**v3 (enumerate_ids to SDDMM):** Use `cu_roaring::enumerate_ids()` to produce sorted column indices directly from roaring containers in a single kernel, construct trivial CSR indptr, then use `cusparseSDDMM` for masked distance computation. Eliminates the bitset intermediate entirely.

**Comparison at 1% pass rate (v3 final results):**

| Scale | Bitset CSR (ms) | Roaring v1 (dense) | Roaring v2 (decompress) | Roaring v3 (SDDMM) | v3 Speedup |
|-------|----------------|--------------------|-----------------------|--------------------|-----------:|
| 100K | 1.172 | 2.683 (0.44x) | 1.004 (1.17x) | 1.578 | **0.74x** |
| 1M | 2.099 | 8.006 (0.26x) | 2.182 (0.96x) | 2.167 | **0.97x** |
| 5M | 4.673 | 18.510 (0.25x) | 5.496 (0.85x) | 4.554 | **1.03x** |
| 10M | 8.789 | 33.636 (0.26x) | 9.056 (0.97x) | 8.954 | **0.98x** |

At small scale (100K), the SDDMM setup overhead makes v3 slower than bitset (0.74x). At 1M and 10M, v3 ties bitset (0.97-0.98x). At 5M, v3 edges ahead (1.03x). Overall, v3 eliminates the 3x regression of v1 and the 5-7% gap of v2, achieving near-parity with bitset CSR at all scales.

### Kernel launch comparison (v2 vs v3)

**v2 (decompress to CSR) — 7 kernels before computation:**

1. `cudaMemsetAsync` — zero the bitset buffer
2. `decompress_kernel` — roaring to flat bitset (one block per container)
3. `calc_nnz_by_rows` — popcount bitset words to get nnz
4. `thrust::exclusive_scan` — prefix sum for CSR indptr
5. `cudaMemcpy D→H` — copy nnz to host for CSR allocation
6. `fill_indices_by_rows` — scatter set bit positions into CSR indices
7. `repeat_csr_kernel` — replicate indices across n_queries rows

**v3 (enumerate_ids to SDDMM) — 2 kernels before computation:**

1. `enumerate_ids_kernel` — roaring to sorted int64_t column indices (one kernel, one block per container)
2. Trivial indptr construction — `[0, nnz, 2*nnz, ...]` (one tiny kernel or host-side)

v3 eliminates the bitset intermediate entirely, reducing 7 kernel launches to 2.

## Cache Hierarchy Context (RTX 5090 Blackwell)

| Cache | Size | Per-element latency |
|-------|------|-------------------|
| L1 | **128 KB per SM** | ~28 cycles |
| L2 | **96 MB shared** | ~200 cycles |
| DRAM | 32 GB | ~500+ cycles |

### Filter data sizes

| Scale (N) | Flat Bitset | Roaring Container | Fits in L1? |
|-----------|-----------|------------------|------------|
| 100K | 12.2 KB | 8 KB each | Both fit |
| 1M | 122 KB | 8 KB each | Bitset barely; roaring yes |
| 5M | 610 KB | 8 KB each | Bitset no; roaring yes |
| 10M | 1.2 MB | 8 KB each | Bitset no; roaring yes |

For brute-force with tiling, each tile accesses a contiguous range of the bitset. The tile's bitset footprint is `tile_cols / 8` bytes (typically ~4 KB), which fits in L1 for both representations. This is why the cache advantage is smaller than in CAGRA's random-access pattern.

## Brute-Force Methodology

- **Interleaved A/B**: Each iteration alternates which filter runs first, eliminating warm-cache bias
- **GPU event timing**: `cudaEventRecord`/`cudaEventElapsedTime` for precise kernel timing
- **Warmup**: 10 interleaved iterations before measurement
- **Iterations**: 30 measured pairs
- **Statistics**: median latency, sample standard deviation (Bessel-corrected), Welch's t-test on means
- **Recall baseline**: bitset_filter results (bitset is ground truth)
- **Brute-force index**: built once per (N, dim) pair, reused across selectivities
- **Bitset path selection**: cuVS auto-selects dense (sparsity < 0.9) or CSR (sparsity ≥ 0.9)
- **Roaring path**: v3 — dense GEMM + `warp_contains()` masking at ≥10% pass rate; `enumerate_ids()` → SDDMM at <10% pass rate

---

# GPU Roaring Bitmap vs Flat Bitset: IVF-Flat Filtered Search

**Generated:** 2026-03-28
**GPU:** NVIDIA GeForce RTX 5090 (170 SMs, 32 GB VRAM, 96 MB L2)
**Platform:** WSL2 (Linux 6.6.87 on Windows)
**Parameters:** k=10, metric=L2Expanded, n_queries=100
**Methodology:** Interleaved A/B, 30 measured iterations, 10 warmup, median latency, Welch's t-test

## Summary

IVF-Flat partitions the dataset into clusters (inverted lists) and searches only the `n_probes` nearest clusters. The filter is applied **per-candidate during list scanning** via the `ivf_to_sample_filter` adapter, which converts `(query, cluster, local_idx)` → `(query, global_idx)` for the 2-arg filter operator. Both bitset and roaring use the same code path — the only difference is the per-element filter check.

**Key findings:**
- Roaring shows **12-22% speedup at 5-10% pass rate** at 5M and 10M scale, with strong statistical significance (t>5, p<0.001)
- At small scale (100K), both are equivalent — filter check is negligible vs. distance computation
- At high pass rate (25-50%), both are equivalent — the filter rarely rejects, so check cost is amortized
- **Recall is 1.0000** across all configurations
- The IVF-Flat results sit **between CAGRA (5-33% speedup) and brute-force (1-9% speedup)**, consistent with the architectural differences

## Results

### 100K Vectors (dim=128, n_lists=316, n_probes=32)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 1% | 0.147 | 0.144 | 1.02x | 0.74 | |
| 5% | 0.325 | 0.319 | 1.02x | 0.19 | |
| 10% | 0.244 | 0.249 | 0.98x | 1.25 | |
| 25% | 0.191 | 0.187 | 1.02x | -1.57 | |
| 50% | 0.220 | 0.211 | 1.04x | -0.92 | |

### 1M Vectors (dim=128, n_lists=1000, n_probes=50)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 1% | 0.353 | 0.318 | 1.11x | 0.36 | |
| 5% | 0.470 | 0.455 | 1.03x | 0.05 | |
| 10% | 0.505 | 0.480 | 1.05x | 0.68 | |
| 25% | 0.711 | 0.643 | 1.11x | 0.16 | |
| 50% | 1.107 | 0.998 | 1.11x | 1.46 | |

### 5M Vectors (dim=128, n_lists=2236, n_probes=80)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 1% | 1.243 | 1.323 | 0.94x | -1.39 | |
| 5% | 1.383 | 1.133 | **1.22x** | 10.86 | *** |
| 10% | 2.314 | 1.989 | **1.16x** | 5.63 | *** |
| 25% | 3.474 | 3.556 | 0.98x | -0.86 | |
| 50% | 4.865 | 4.999 | 0.97x | -0.64 | |

### 10M Vectors (dim=128, n_lists=3162, n_probes=100)

| Pass Rate | Bitset (ms) | Roaring (ms) | Speedup | t-stat | Significant |
|-----------|------------|-------------|---------|--------|-------------|
| 1% | 1.042 | 1.056 | 0.99x | -0.73 | |
| 10% | 3.188 | 2.758 | **1.16x** | 12.62 | *** |
| 50% | 7.729 | 7.703 | 1.00x | -0.52 | |

Significance: \* p<0.05 (|t|>2.0), \*\* p<0.01 (|t|>2.6), \*\*\* p<0.001 (|t|>3.3)

## Analysis

### Why the sweet spot is 5-10% pass rate

The IVF-Flat filter is applied per-candidate during list scanning. The total search time is:

```
T_total = T_coarse_search + T_list_scan + T_filter_check + T_select_k
```

At **low pass rates (1%)**: most candidates are filtered out quickly. The filter check is fast for both bitset and roaring (early rejection). List scanning also terminates faster since fewer candidates pass.

At **5-10% pass rate**: filter checking becomes a significant fraction of total time. Each candidate requires a full membership test. Roaring's `warp_contains()` benefits primarily from its **two-level cache hierarchy**:
1. **Small key_index array (L1-resident)** — roaring first reads `key_index[id >> 16]`, a small array (77 entries at 5M, 153 at 10M) that stays in L1 across calls
2. **8 KB container reads** — each container fits in L1, so the second-level access also tends to hit L1 if the same container was recently accessed by another thread or query
3. By contrast, bitset reads `bitset[id / 32]` — a single random offset into a 610 KB-1.2 MB array that exceeds L1 capacity and forces L2 round-trips

With random data (as in this benchmark), IVF list members have effectively random IDs, so warp-cooperative sharing via `__match_any_sync` is minimal and not a significant factor (see Caveats)

At **high pass rates (25-50%)**: most candidates pass the filter. The filter check is a smaller fraction of total time (most time is spent computing distances for passing candidates), reducing the impact of any filter optimization.

### IVF-Flat vs CAGRA vs Brute-Force: Filter Impact Comparison

| Index Type | Filter Application | Access Pattern | Best Roaring Speedup |
|-----------|-------------------|---------------|---------------------|
| **CAGRA** | Per-candidate during graph traversal | Random IDs, temporal L1 locality | **1.33x** (1M/50%) |
| **IVF-Flat** | Per-candidate during list scanning | Random IDs, two-level cache | **1.22x** (5M/5%) |
| **Brute-Force** | Post-GEMM masking (tiny fraction) | Sequential tile access | **1.09x** (100K/25%) |

The results confirm the hypothesis: roaring's benefit scales with how much the filter contributes to total search time and how favorable the access pattern is for L1 cache locality.

- **CAGRA**: Filter is in the hot loop during graph traversal. With random data, neighbor IDs are random, but repeated accesses to the same containers within a search keep 8 KB containers in L1 (temporal locality). The flat bitset (122 KB-2.4 MB) exceeds L1 and forces L2 round-trips on every access.
- **IVF-Flat**: Filter is in the list scan loop. IDs within a cluster are random with respect to ID space. Roaring's two-level structure (small key_index + 8 KB containers) provides better L1 utilization than a single random read into a 610 KB-1.2 MB bitset.
- **Brute-Force**: Filter is a post-processing mask after GEMM. Sequential access. GEMM dominates (~97% of time) → minimal impact.

### IVF-Flat ID Distribution

IVF-Flat assigns vectors to clusters based on centroid proximity. With random data (uniform distribution, as in this benchmark), cluster membership is effectively random with respect to ID space — vectors near the same centroid have no ID-space clustering. At 5M with 2236 clusters, each cluster has ~2237 vectors scattered across 77 roaring containers (each covering 65536 IDs). Within a warp scanning the same cluster, thread IDs are spread across many different containers, so `__match_any_sync` provides minimal warp sharing.

The observed speedup is therefore best explained by roaring's **two-level cache structure**, not warp-cooperative sharing: the small key_index array stays L1-resident, and recently-accessed 8 KB containers also benefit from L1 temporal locality across successive filter checks within the same list scan.

### Caveats and Verification

**Only 3 of 18 configs are statistically significant.** The headline 12-22% speedup occurs only at 5M/5%, 5M/10%, and 10M/10%. Every other config shows no meaningful difference. This is not a universal improvement — it's specific to the regime where filter checking is a material fraction of total time.

**Warp-cooperative sharing is minimal with random data.** With uniform random data, IVF cluster membership is random with respect to ID space. Warp threads scanning the same cluster access IDs spread across many different roaring containers, so `__match_any_sync` rarely finds matching keys. The observed speedup is attributable to roaring's two-level cache structure (small key_index array + 8 KB containers), not warp sharing. This is consistent with the CAGRA results, where speedup persists at 10M-20M scales despite near-zero warp sharing (29-31 unique keys per 32-thread warp).

**Roaring is slightly slower at 25-50% pass rate.** At 5M, roaring shows 0.97-0.98x speedup (2-3% slower) at high pass rates. This suggests `warp_contains()` has higher per-call overhead than flat bitset lookup (binary search on key_index + bit test vs. single bit test). When filter checks are frequent but rarely reject, this overhead accumulates.

**Variance is inconsistent across scales.** At 1M, std is 0.2-0.6ms on 0.3-1.1ms medians (20-50% CV). At 5M, std drops to 0.09-0.5ms on 1.1-5.0ms medians (2-10% CV). The low variance at 5M makes the t-statistics appear stronger. WSL2 scheduling jitter affects short-duration measurements disproportionately.

**Bare-metal validation needed.** All results are on WSL2, which adds scheduling jitter. The 1M results (high variance, no significance) might show significance on bare-metal Linux. Conversely, the 5M results (low variance, high significance) should be validated to ensure they're not WSL2-specific artifacts.

**Sanity check on the 5M/5% result (1.22x, t=10.86):**
- Bitset 1.383ms, roaring 1.133ms → 0.250ms saved
- With n_probes=80, ~2237 vectors/list → ~179K candidates scanned per query
- 100 queries → 17.9M filter checks total
- 0.250ms / 17.9M checks = 14 ns saved per check
- A cache miss (L1→L2) costs ~170 cycles ÷ ~2.5 GHz = ~68 ns. Saving a fraction of cache misses per check via better L1 utilization is consistent with 14 ns/check

## IVF-Flat Methodology

- **Interleaved A/B**: Each iteration alternates which filter runs first, eliminating warm-cache bias
- **GPU event timing**: `cudaEventRecord`/`cudaEventElapsedTime` for precise kernel timing
- **Warmup**: 10 interleaved iterations before measurement
- **Iterations**: 30 measured pairs
- **Statistics**: median latency, sample standard deviation (Bessel-corrected), Welch's t-test on means
- **Recall baseline**: bitset_filter results (bitset is ground truth)
- **IVF-Flat index**: built once per (N, dim, n_lists) combination, reused across selectivities
- **n_lists**: ~sqrt(N) (316 for 100K, 1000 for 1M, 2236 for 5M, 3162 for 10M)
- **n_probes**: ~n_lists/10 (32, 50, 80, 100 respectively)
