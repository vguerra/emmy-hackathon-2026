#!/usr/bin/env python3
"""Compile each assemble-only problem with compile.py and check that the runner's ptxas accepts the PTX.

    python tools/check_assembles.py problems_asm

Execution is not attempted. Needs network access to the runner.
"""

import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://runner.riftstack.ai:48701"


def check(problem: Path) -> tuple[bool, str]:
    meta = json.loads((problem / "meta.json").read_text())
    for kernel in meta["kernels"]:
        ir = next(problem.glob(f"{kernel['entry']}.ir.json"))
        compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
        if compiled.returncode != 0:
            return False, compiled.stderr.strip().splitlines()[-1]
        reply = requests.post(f"{RUNNER}/compile", json={"ptx": compiled.stdout}, timeout=120).json()
        if not reply.get("ok"):
            return False, "ptxas: " + reply.get("stderr", "").strip().splitlines()[0]
        registers = reply.get("registers", {}).get(kernel["entry"])
    return True, f"[{kernel['entry']}] ptxas accepted ({registers} registers)"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "problems_asm")
    failed = 0
    for problem in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        ok, detail = check(problem)
        failed += not ok
        print(f"{'ASSEMBLES' if ok else 'FAIL'}  {problem.name:<24} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
