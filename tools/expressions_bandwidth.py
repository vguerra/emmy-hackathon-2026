"""Large memory-bound problems for measuring achieved bandwidth (see tools/bandwidth.py).

Too big to commit (the reference outputs alone are tens of MB); generate them outside the repo:

    python tools/generate.py tools/expressions_bandwidth.py --emmy .venv/bin/emmy --out /tmp/bw
"""

N = 1 << 24                                           # 16.8M f32 = 64 MiB per buffer, well past the 5090's L2

OPEN = [
    dict(id="bw_add", code=f"torch.randn({N}) + torch.randn({N})",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="bw_scale", code=f"torch.randn({N}) * 2.0 + 1.0",
         ref=lambda x: [x * 2.0 + 1.0], rtol=1e-6, atol=1e-6),
    dict(id="bw_sum", code=f"torch.randn({N}).sum()",
         ref=lambda x: [x.astype("float64").sum().astype("float32")], rtol=1e-3, atol=1e-1),
    dict(id="bw_row_sum", code="torch.randn(8192, 4096).sum(-1)",
         ref=lambda x: [x.astype("float64").sum(-1).astype("float32")], rtol=1e-3, atol=1e-2),
]
