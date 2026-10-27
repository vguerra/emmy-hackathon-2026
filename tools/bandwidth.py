#!/usr/bin/env python3
"""Achieved memory bandwidth of memory-bound kernels, against the card's peak.

    python tools/generate.py tools/expressions_bandwidth.py --emmy .venv/bin/emmy --out /tmp/bw
    python tools/bandwidth.py /tmp/bw

Bytes moved = every buffer argument read or written once (the minimum traffic); time = warm median from
tools/bench.py. The RTX 5090's peak is about 1792 GB/s; pass --peak for another card. A buffer that fits the
L2 (96 MiB on the 5090) can exceed the peak because warm launches find it cached; the default problems are
sized past that.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import bench                                   # noqa: E402

DTYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i32": 4, "i64": 8}


def traffic_bytes(problem: Path) -> int:
    kernel = json.loads((problem / "meta.json").read_text())["kernels"][0]
    total = 0
    for arg in kernel["args"]:
        count = 1
        for extent in arg["shape"]:
            count *= extent
        total += count * DTYPE_BYTES[arg["dtype"]]
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="a folder of problem folders")
    parser.add_argument("--peak", type=float, default=1792.0, help="peak DRAM bandwidth in GB/s")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    print(f"{'problem':16}{'MB moved':>10}{'us':>10}{'GB/s':>9}{'% of peak':>11}")
    for problem in sorted(p for p in Path(args.folder).iterdir() if (p / "meta.json").exists()):
        micros = bench(problem, 10, args.iters)
        if micros is None:
            continue
        moved = traffic_bytes(problem)
        rate = moved / micros / 1e3                       # bytes per microsecond -> GB/s
        print(f"{problem.name:16}{moved / 1e6:>10.1f}{micros:>10.1f}{rate:>9.0f}{100 * rate / args.peak:>10.0f}%")


if __name__ == "__main__":
    main()
