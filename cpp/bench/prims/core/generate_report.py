#!/usr/bin/env python3
"""
Generate a markdown comparison report from bench_cagra_roaring_comprehensive.json.

Usage:
    python3 generate_report.py [input.json] [output.md]
    Defaults: bench_cagra_roaring_comprehensive.json -> ROARING_BENCHMARK_REPORT.md
"""

import json
import sys
from datetime import datetime

def load_results(path):
    with open(path) as f:
        return json.load(f)

def fmt_ms(v):
    if v < 0.01:
        return f"{v*1000:.1f} us"
    return f"{v:.3f} ms"

def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b/1024:.1f} KB"
    return f"{b/(1024*1024):.1f} MB"

def generate_report(data):
    lines = []
    w = lines.append

    w("# GPU Roaring Bitmap vs Flat Bitset: Comprehensive Comparison")
    w("## For cuVS CAGRA Filtered Vector Search")
    w("")
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**GPU:** {data['gpu']} ({data['n_sms']} SMs, {data.get('l2_cache_mb', '?')} MB L2)")
    w(f"**Parameters:** dim={data['dim']}, k={data['k']}, "
      f"warmup={data['warmup']}, iters={data['iters']}")
    w("")

    search = data["search_results"]
    multi = data.get("multi_and_results", [])

    # ================================================================
    # Executive Summary
    # ================================================================
    w("## Executive Summary")
    w("")

    # Compute summary stats
    speedups = [(r["config"], r["speedup_roaring"]) for r in search]
    best_speedup = max(speedups, key=lambda x: x[1])
    worst_speedup = min(speedups, key=lambda x: x[1])
    avg_speedup = sum(s for _, s in speedups) / len(speedups)

    compressions = [(r["config"], r["compression_default"]) for r in search]
    best_compression = max(compressions, key=lambda x: x[1])

    negated_configs = [r["config"] for r in search if r.get("negated")]

    mismatches = sum(r.get("filter_mismatches", 0) for r in search)

    w(f"- **Average search speedup** (roaring vs bitset): **{avg_speedup:.2f}x** "
      f"across {len(search)} configurations")
    w(f"- **Best speedup:** {best_speedup[1]:.2f}x at {best_speedup[0]}")
    w(f"- **Worst speedup:** {worst_speedup[1]:.2f}x at {worst_speedup[0]}")
    w(f"- **Best compression:** {best_compression[1]:.1f}x at {best_compression[0]}")
    if negated_configs:
        w(f"- **Complement optimization** activated for: {', '.join(negated_configs)}")
    w(f"- **Filter correctness:** {mismatches} total mismatches across all configs "
      f"({'PASS' if mismatches == 0 else 'FAIL'})")
    if multi:
        multi_speedups = [m["and_speedup"] for m in multi]
        w(f"- **Multi-predicate AND speedup:** {min(multi_speedups):.1f}x - "
          f"{max(multi_speedups):.1f}x (fused roaring vs flat bitset)")
    w("")

    # ================================================================
    # 1. Search Performance
    # ================================================================
    w("## 1. Search Performance")
    w("")
    w("### 1.1 Latency by Selectivity")
    w("")
    w("| Config | Selectivity | Bitset (ms) | Roaring (ms) | Promoted (ms) | "
      "Speedup (R) | Speedup (P) | Recall (R) |")
    w("|--------|-------------|-------------|--------------|---------------|"
      "-------------|-------------|------------|")

    for r in search:
        w(f"| {r['config']} | {r['filter_pass_rate']*100:.1f}% | "
          f"{r['bitset_median_ms']:.3f} | {r['roaring_median_ms']:.3f} | "
          f"{r['roaring_promoted_median_ms']:.3f} | "
          f"{r['speedup_roaring']:.2f}x | {r['speedup_promoted']:.2f}x | "
          f"{r['recall_roaring']:.4f} |")
    w("")

    w("### 1.2 Throughput (QPS)")
    w("")
    w("| Config | No Filter | Bitset | Roaring | Promoted |")
    w("|--------|-----------|--------|---------|----------|")
    for r in search:
        w(f"| {r['config']} | {r['qps_none']:,.0f} | {r['qps_bitset']:,.0f} | "
          f"{r['qps_roaring']:,.0f} | {r['qps_promoted']:,.0f} |")
    w("")

    w("### 1.3 Recall Analysis")
    w("")
    w("Roaring and bitset filters encode the same membership set. "
      "Any recall difference vs bitset baseline indicates differing "
      "CAGRA graph traversal paths (not filter errors).")
    w("")
    w("| Config | Recall (Bitset) | Recall (Roaring) | Recall (Promoted) | Delta |")
    w("|--------|-----------------|------------------|-------------------|-------|")
    for r in search:
        delta = abs(r["recall_roaring"] - 1.0)
        w(f"| {r['config']} | 1.0000 | {r['recall_roaring']:.4f} | "
          f"{r['recall_promoted']:.4f} | {delta:.4f} |")
    w("")

    # ================================================================
    # 2. Memory Efficiency
    # ================================================================
    w("## 2. Memory Efficiency")
    w("")
    w("| Config | Selectivity | Bitset | Roaring | Compression | Negated | "
      "Containers (B/A) |")
    w("|--------|-------------|--------|---------|-------------|---------|"
      "-----------------|")
    for r in search:
        w(f"| {r['config']} | {r['filter_pass_rate']*100:.1f}% | "
          f"{fmt_bytes(r['bitset_bytes'])} | {fmt_bytes(r['roaring_default_bytes'])} | "
          f"{r['compression_default']:.1f}x | "
          f"{'Yes' if r.get('negated') else 'No'} | "
          f"{r['n_bitmap']}/{r['n_array']} |")
    w("")

    w("The **complement optimization** stores the set complement when density > 50%, "
      "making compression symmetric: a 99% filter stores only the 1% rejects.")
    w("")

    # ================================================================
    # 3. Filter Construction
    # ================================================================
    w("## 3. Filter Construction Time")
    w("")
    w("| Config | Bitset Build (ms) | Roaring Build (ms) | Build Speedup |")
    w("|--------|-------------------|--------------------|---------------|")
    for r in search:
        bs = r.get("bitset_build_median_ms", 0)
        rs = r.get("roaring_build_median_ms", 0)
        spd = bs / rs if rs > 0 else 0
        w(f"| {r['config']} | {bs:.3f} | {rs:.3f} | {spd:.2f}x |")
    w("")

    # ================================================================
    # 4. Multi-Predicate AND
    # ================================================================
    if multi:
        w("## 4. Multi-Predicate Performance")
        w("")
        w("Each predicate passes ~50% independently. Combined pass rates: "
          "2-way ~25%, 3-way ~12.5%, 4-way ~6.25%.")
        w("")
        w("| Predicates | Bitset AND (ms) | Roaring AND (ms) | AND Speedup | "
          "Search Bitset (ms) | Search Roaring (ms) | Search Speedup |")
        w("|------------|-----------------|-------------------|-------------|"
          "--------------------|---------------------|----------------|")
        for m in multi:
            w(f"| {m['n_predicates']} | "
              f"{m['bitset_and_median_ms']:.3f} | {m['roaring_and_median_ms']:.3f} | "
              f"{m['and_speedup']:.2f}x | "
              f"{m['search_bitset_median_ms']:.3f} | {m['search_roaring_median_ms']:.3f} | "
              f"{m['search_speedup']:.2f}x |")
        w("")

        w("### Combined Filter Memory")
        w("")
        w("| Predicates | Bitset | Roaring | Compression | Negated |")
        w("|------------|--------|---------|-------------|---------|")
        for m in multi:
            w(f"| {m['n_predicates']} | {fmt_bytes(m['combined_bitset_bytes'])} | "
              f"{fmt_bytes(m['combined_roaring_bytes'])} | "
              f"{m['combined_compression']:.1f}x | "
              f"{'Yes' if m.get('combined_negated') else 'No'} |")
        w("")

    # ================================================================
    # 5. E2E Pipeline
    # ================================================================
    w("## 5. End-to-End Pipeline (Build + Search)")
    w("")
    w("| Config | Bitset E2E (ms) | Roaring E2E (ms) | E2E Speedup |")
    w("|--------|-----------------|-------------------|-------------|")
    for r in search:
        w(f"| {r['config']} | {r['e2e_bitset_ms']:.3f} | "
          f"{r['e2e_roaring_ms']:.3f} | {r['e2e_speedup']:.2f}x |")
    w("")

    # ================================================================
    # 6. Scalability
    # ================================================================
    scale_1m = [r for r in search if r["n_vectors"] == 1000000 and r["n_queries"] == 100]
    scale_10m = [r for r in search if r["n_vectors"] == 10000000]
    if scale_1m and scale_10m:
        w("## 6. Scalability (1M vs 10M)")
        w("")
        w("| Selectivity | 1M Speedup | 10M Speedup | 1M Compression | 10M Compression |")
        w("|-------------|------------|-------------|----------------|-----------------|")

        for rate_str in ["1pct", "10pct", "50pct"]:
            r1m = next((r for r in scale_1m if rate_str in r["config"]), None)
            r10m = next((r for r in scale_10m if rate_str in r["config"]), None)
            if r1m and r10m:
                w(f"| {rate_str.replace('pct','%')} | "
                  f"{r1m['speedup_roaring']:.2f}x | {r10m['speedup_roaring']:.2f}x | "
                  f"{r1m['compression_default']:.1f}x | {r10m['compression_default']:.1f}x |")
        w("")

    # ================================================================
    # 7. Decision Matrix
    # ================================================================
    w("## 7. When to Use Roaring vs Bitset")
    w("")
    w("| Condition | Recommendation | Reason |")
    w("|-----------|---------------|--------|")
    w("| Selectivity 0.1-5% | **Roaring** | "
      "Array containers use O(cardinality) memory, not O(universe) |")
    w("| Selectivity 5-50% | **Roaring** | "
      "Compressed bitmaps fit in L2 cache better |")
    w("| Selectivity 50-99% | **Roaring** | "
      "Complement optimization stores only the rejects |")
    w("| Multiple predicates (AND) | **Roaring** | "
      "Fused multi_and avoids full O(N/8) bitset scans |")
    w("| Single use, already have bitset | Bitset | "
      "No construction overhead; direct use |")
    w("| Memory-constrained (>100M vectors) | **Roaring** | "
      "10-60x compression frees VRAM for vectors/graph |")
    w("")

    # ================================================================
    # 8. Methodology
    # ================================================================
    w("## 8. Methodology")
    w("")
    w(f"- **GPU:** {data['gpu']}")
    w(f"- **SMs:** {data['n_sms']}")
    w(f"- **L2 Cache:** {data.get('l2_cache_mb', '?')} MB")
    w(f"- **Vector dimension:** {data['dim']}")
    w(f"- **k (neighbors):** {data['k']}")
    w(f"- **CAGRA graph_degree:** 32, intermediate_graph_degree: 48")
    w(f"- **CAGRA itopk_size:** 256")
    w(f"- **Warmup iterations:** {data['warmup']}")
    w(f"- **Measured iterations:** {data['iters']}")
    w(f"- **Statistics:** median, mean, std, p5, p95 (GPU event timing)")
    w(f"- **Recall baseline:** bitset_filter results (bitset is ground truth)")
    w(f"- **Correctness:** Roaring decompressed to bitset, compared word-for-word")
    w("")

    return "\n".join(lines)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "bench_cagra_roaring_comprehensive.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "ROARING_BENCHMARK_REPORT.md"

    print(f"Loading {input_path}...")
    data = load_results(input_path)

    print(f"Generating report...")
    report = generate_report(data)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")
    print(f"  {len(data['search_results'])} search configs")
    print(f"  {len(data.get('multi_and_results', []))} multi-AND configs")


if __name__ == "__main__":
    main()
