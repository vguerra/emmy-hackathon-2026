#!/usr/bin/env python3
"""Run compile.py through submit.py on every problem folder under the given directories.

    python tools/run_all.py                        # problems/ and problems_extra/
    python tools/run_all.py problems               # one directory
    python tools/run_all.py problems_extra -k softmax

Prints one line per problem (PASS/FAIL plus submit.py's last line) and a summary.
Exit code 0 means every problem passed. Needs network access to the runner.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# submit.py sends each intermediate buffer inline as JSON, which the runner refuses past this many values
MAX_INLINE_VALUES = 500_000


def problem_dirs(roots: list[str], pattern: str) -> list[Path]:
    found = []
    for root in roots:
        base = (ROOT / root) if not Path(root).is_absolute() else Path(root)
        found += sorted(d for d in base.iterdir() if d.is_dir() and (d / "meta.json").exists())
    return [d for d in found if pattern in d.name]


def has_large_intermediate(problem: Path) -> bool:
    """True for a multi-kernel problem whose chained buffer is too big for submit.py to send."""
    kernels = json.loads((problem / "meta.json").read_text())["kernels"]
    return len(kernels) > 1 and any(
        a.get("intermediate") and math.prod(a["shape"]) > MAX_INLINE_VALUES
        for k in kernels for a in k["args"])


def run_problem(problem: Path) -> tuple[bool, str]:
    if has_large_intermediate(problem):                 # check the first kernel and reduce its partials here
        script = [str(ROOT / "tools" / "check_partial.py"), str(problem)]
    else:
        script = [str(ROOT / "submit.py"), str(problem), "--compiler", str(ROOT / "compile.py")]
    result = subprocess.run([sys.executable, *script], capture_output=True, text=True)
    lines = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else "(no output)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", default=["problems", "problems_extra"],
                        help="directories holding problem folders")
    parser.add_argument("-k", default="", help="only problems whose name contains this text")
    args = parser.parse_args()

    problems = problem_dirs(args.dirs, args.k)
    failed = []
    for problem in problems:
        ok, detail = run_problem(problem)
        print(f"{'PASS' if ok else 'FAIL'}  {problem.name:<24} {detail[:110]}", flush=True)
        if not ok:
            failed.append(problem.name)

    print(f"\n{len(problems) - len(failed)}/{len(problems)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
