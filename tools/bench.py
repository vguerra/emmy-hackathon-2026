#!/usr/bin/env python3
"""Time the compiled kernels on the runner: median of many launches after a warmup.

submit.py times a single cold launch, which is dominated by launch overhead for small kernels.
This runs each single-kernel problem with warmup and many timed iterations.

    python tools/bench.py problems/m01_matmul_512 problems_extra/y_a_256x64
    python tools/bench.py problems --json before.json     # save, then compare two runs:
    python tools/bench.py problems --compare before.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://runner.riftstack.ai:48701"


def bench(problem: Path, warmup: int, iters: int) -> float | None:
    meta = json.loads((problem / "meta.json").read_text())
    if len(meta["kernels"]) != 1:
        return None                                  # chained kernels need submit.py's plumbing
    kernel = meta["kernels"][0]
    ir = next(problem.glob(f"{kernel['entry']}.ir.json"))
    ptx = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True, check=True).stdout
    compiled = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=120).json()
    if not compiled.get("ok"):
        raise RuntimeError(compiled.get("stderr", "ptxas failed"))
    reply = requests.post(f"{RUNNER}/run", json={
        "cubin_id": compiled["cubin_id"], "entry": kernel["entry"], "grid": kernel["grid"], "block": kernel["block"],
        "smem_bytes": kernel["smem_bytes"], "zero_outputs": kernel["zero_outputs"], "args": kernel["args"],
        "warmup": warmup, "iters": iters, "max_return": 1}, timeout=300).json()
    return reply["median_us"] if reply.get("ok") else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="problem folders, or folders of problem folders")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", help="write results to this file")
    parser.add_argument("--compare", help="a --json file from an earlier run; print the ratio")
    args = parser.parse_args()

    problems = []
    for path in map(Path, args.paths):
        path = path if path.is_absolute() else ROOT / path
        problems += [path] if (path / "meta.json").exists() else sorted(p for p in path.iterdir() if (p / "meta.json").exists())
    before = json.loads(Path(args.compare).read_text()) if args.compare else {}
    results = {}
    for problem in problems:
        us = bench(problem, args.warmup, args.iters)
        if us is None:
            continue
        results[problem.name] = us
        ratio = f"  x{before[problem.name] / us:.2f}" if problem.name in before else ""
        print(f"{problem.name:<24} {us:10.1f} us{ratio}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
