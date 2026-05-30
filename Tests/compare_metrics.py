#!/usr/bin/env python3
"""Print CPU vs GPU table from cpu_bench.out and gpu_bench.out."""
import re
from pathlib import Path


def load_metrics(path):
    data = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(
            r"BENCH_METRIC backend=(\w+) case=(\d+) metric=(\w+) ms=([\d.]+)",
            line,
        )
        if m:
            backend, case, metric, ms = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
            data.setdefault(case, {})[f"{backend}_{metric}"] = ms
    return data


def main():
    cpu = load_metrics("cpu_bench.out")
    gpu = load_metrics("gpu_bench.out")
    cases = sorted(set(cpu) | set(gpu))

    print()
    print("=== CPU vs GPU (ms) ===")
    print(f"{'case':>4}  {'cpu_hash':>10}  {'gpu_upload':>10}  {'gpu_hash':>10}  {'gpu_copy':>10}  {'gpu_total':>10}  {'speedup':>8}")
    print("-" * 72)

    for c in cases:
        ch = cpu.get(c, {}).get("cpu_hash")
        gu = gpu.get(c, {}).get("gpu_upload")
        gh = gpu.get(c, {}).get("gpu_hash")
        gc = gpu.get(c, {}).get("gpu_copy")
        gt = gpu.get(c, {}).get("gpu_total")
        speedup = (ch / gt) if ch and gt and gt > 0 else None

        def fmt(v):
            return f"{v:10.3f}" if v is not None else f"{'—':>10}"

        sp = f"{speedup:8.2f}x" if speedup is not None else f"{'—':>8}"
        print(f"{c:4d}  {fmt(ch)}  {fmt(gu)}  {fmt(gh)}  {fmt(gc)}  {fmt(gt)}  {sp}")


if __name__ == "__main__":
    main()
