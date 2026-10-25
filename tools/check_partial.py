"""Run only the first kernel of a two-kernel split-K problem and reduce its partials locally.

submit.py chains kernels by sending each intermediate buffer inline as JSON, which the runner
refuses when the partial buffer is millions of values. This runs kernel 1 through the runner
API, sums the partials over the split axis here, and compares with expected.npz.

    python tools/check_partial.py problems_extra/z_h_matmul_rect
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, requests

root = Path(__file__).resolve().parent.parent
pdir = (root / sys.argv[1]) if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
meta = json.loads((pdir / "meta.json").read_text())
expected = dict(np.load(pdir / "expected.npz"))
k = meta["kernels"][0]
ptx = subprocess.run([sys.executable, str(root / "compile.py"), str(next(pdir.glob(f"{k['entry']}.ir.json")))],
                     capture_output=True, text=True, check=True).stdout
server = "http://runner.riftstack.ai:48701"
c = requests.post(f"{server}/compile", json={"ptx": ptx}).json()
assert c["ok"], c
need = int(np.prod(k["args"][-1]["shape"]))
r = requests.post(f"{server}/run", json={"cubin_id": c["cubin_id"], "entry": k["entry"], "grid": k["grid"],
    "block": k["block"], "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": k["args"],
    "warmup": 0, "iters": 1, "max_return": need}, timeout=300).json()
assert r["ok"], r
out = k["args"][-1]
part = np.array(r["outputs"][out["name"]]["values"], dtype=np.float32).reshape(out["shape"])
got = part.sum(0).astype(np.float16)
want = list(expected.values())[0]
err = np.abs(got.astype(np.float64) - want.astype(np.float64)).max()
print(f"[{k['entry']}] PASS  partial {part.shape} reduced locally: max abs err {err:.3e}" if err <= meta["atol"] + meta["rtol"] * np.abs(want.astype(np.float64)).max() else f"[{k['entry']}] FAIL  max abs err {err:.3e}")
tolerance = meta["atol"] + meta["rtol"] * np.abs(want.astype(np.float64)).max()
sys.exit(0 if err <= tolerance else 1)
