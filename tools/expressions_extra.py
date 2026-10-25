"""Extra probe problems: same categories as the open set, different shapes and ops."""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
f64 = lambda a: a.astype(np.float64)
f32 = lambda a: a.astype(np.float32)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    # ---- pointwise ---------------------------------------------------------
    dict(id="x_p_sub", code="torch.randn(2048) - torch.randn(2048)",
         ref=lambda a, b: [a - b], rtol=0, atol=0),
    dict(id="x_p_div", code="x = torch.randn(1024); y = torch.randn(1024); x / (y * y + 1.0)",
         ref=lambda x, y: [x / (y * y + np.float32(1.0))], rtol=1e-5, atol=1e-5),
    dict(id="x_p_exp", code="torch.exp(torch.randn(1024))",
         ref=lambda a: [f32(np.exp(f64(a)))], rtol=1e-4, atol=1e-5),
    dict(id="x_p_tanh", code="torch.tanh(torch.randn(1024))",
         ref=lambda a: [f32(np.tanh(f64(a)))], rtol=1e-4, atol=1e-5),
    dict(id="x_p_sigmoid", code="torch.sigmoid(torch.randn(1024))",
         ref=lambda a: [f32(1 / (1 + np.exp(-f64(a))))], rtol=1e-4, atol=1e-5),
    dict(id="x_p_log", code="x = torch.randn(1024); torch.log(x * x + 0.5)",
         ref=lambda x: [f32(np.log(f64(x) ** 2 + 0.5))], rtol=1e-4, atol=1e-5),
    dict(id="x_p_abs_neg", code="torch.abs(torch.randn(1024)) - torch.randn(1024)",
         ref=lambda a, b: [np.abs(a) - b], rtol=0, atol=0),
    dict(id="x_p_clamp", code="torch.clamp(torch.randn(1024), -0.5, 0.5)",
         ref=lambda a: [np.clip(a, -0.5, 0.5)], rtol=0, atol=0),
    dict(id="x_p_odd", code="torch.randn(3000) * torch.randn(3000) + 2.0",
         ref=lambda a, b: [a * b + np.float32(2.0)], rtol=1e-6, atol=1e-6),
    dict(id="x_p_2d", code="torch.randn(32, 100) * torch.randn(32, 100)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="x_p_bcast", code="torch.randn(64, 128) + torch.randn(128)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="x_p_max2", code="torch.maximum(torch.randn(1024), torch.randn(1024))",
         ref=lambda a, b: [np.maximum(a, b)], rtol=0, atol=0),

    # ---- reductions (pinned to one coop block like the open set) ----------
    dict(id="x_r_max", code="torch.amax(torch.randn(4096))",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="x_r_min", code="torch.neg(torch.amax(torch.neg(torch.randn(2048))))",
         ref=lambda x: [x.min()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="x_r_odd_sum", code="torch.randn(1000).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_row_max", code="torch.randn(32, 300).max(-1).values",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="x_r_row_mean", code="torch.randn(48, 256).mean(-1)",
         ref=lambda x: [f32(f64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_col_sum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [f32(f64(x).sum(0))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_sumsq", code="x = torch.randn(2048); (x * x).sum()",
         ref=lambda x: [f32((f64(x) ** 2).sum())], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),

    # ---- matmul ------------------------------------------------------------
    dict(id="x_m_64", code="torch.randn(64, 64) @ torch.randn(64, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_256x128", code="torch.randn(256, 128) @ torch.randn(128, 256)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_tall", code="torch.randn(512, 64) @ torch.randn(64, 32)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_relu", code="torch.relu(torch.randn(128, 128) @ torch.randn(128, 128))",
         ref=lambda a, b: [np.maximum(f32(f64(a) @ f64(b)), 0)], rtol=1e-3, atol=1e-3),
    dict(id="x_m_bias", code="torch.randn(128, 64) @ torch.randn(64, 128) + torch.randn(128)",
         ref=lambda a, b, c: [f32(f64(a) @ f64(b) + f64(c))], rtol=1e-3, atol=1e-3),

    # ---- fused -------------------------------------------------------------
    dict(id="x_f_softmax_wide", code="torch.softmax(torch.randn(16, 1024), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_softmax_odd", code="torch.softmax(torch.randn(32, 100), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_logsoftmax", code="x = torch.randn(32, 256); x - torch.log(torch.exp(x).sum(-1, keepdim=True))",
         ref=lambda x: [f32(np.log(_softmax(f64(x))))], rtol=1e-3, atol=1e-4),
    dict(id="x_f_center", code="x = torch.randn(32, 256); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) - f64(x).mean(-1, keepdims=True))], rtol=1e-3, atol=1e-4),
    dict(id="x_f_rmsnorm_small", code="x = torch.randn(32, 256); w = torch.randn(256); "
         "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=lambda x, w: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-5) * f64(w))],
         rtol=1e-3, atol=1e-4),
    dict(id="x_f_l2norm_wide", code="x = torch.randn(16, 2048); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=lambda x: [f32(f64(x) / np.sqrt((f64(x) ** 2).sum(-1, keepdims=True)))], rtol=1e-3, atol=1e-4),

    # ---- attention-like ----------------------------------------------------
    dict(id="x_a_32", code="q = torch.randn(32, 32); kt = torch.randn(32, 32); v = torch.randn(32, 32); "
         "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 5.656854) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_128x64", code="q = torch.randn(128, 64); kt = torch.randn(64, 128); v = torch.randn(128, 64); "
         "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_rect", code="q = torch.randn(64, 32); kt = torch.randn(32, 256); v = torch.randn(256, 32); "
         "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 5.656854) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_noscale", code="q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); "
         "torch.softmax(q @ kt, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt)) @ f64(v))], rtol=1e-3, atol=1e-4),
]

# ---- round 2: sizes that are not block multiples, other block sizes, more ops -----------------
T256 = "WORK=t256,REDUCE=coop"
T64 = "WORK=t64,REDUCE=coop"
T32 = "WORK=t32,REDUCE=coop"

OPEN += [
    # pointwise
    dict(id="y_p_tiny", code="torch.randn(100) + torch.randn(100)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="y_p_prime", code="torch.randn(4097) * 2.0",
         ref=lambda a: [a * np.float32(2.0)], rtol=1e-6, atol=1e-6),
    dict(id="y_p_3d", code="torch.randn(8, 16, 32) - torch.randn(8, 16, 32)",
         ref=lambda a, b: [a - b], rtol=0, atol=0),
    dict(id="y_p_const_chain", code="torch.randn(2048) * 3.0 - 1.5",
         ref=lambda a: [a * np.float32(3.0) - np.float32(1.5)], rtol=1e-6, atol=1e-6),
    dict(id="y_p_sqrt_abs", code="torch.sqrt(torch.abs(torch.randn(1024)))",
         ref=lambda a: [np.sqrt(np.abs(a))], rtol=1e-6, atol=1e-6),
    dict(id="y_p_rsqrt", code="torch.rsqrt(torch.randn(1024) ** 2 + 1.0)",
         ref=lambda a: [f32(1 / np.sqrt(f64(a) ** 2 + 1.0))], rtol=1e-5, atol=1e-5),
    dict(id="y_p_silu", code="x = torch.randn(1024); x * torch.sigmoid(x)",
         ref=lambda x: [f32(f64(x) / (1 + np.exp(-f64(x))))], rtol=1e-4, atol=1e-5),
    dict(id="y_p_gauss", code="x = torch.randn(1024); torch.exp(-x * x)",
         ref=lambda x: [f32(np.exp(-f64(x) ** 2))], rtol=1e-4, atol=1e-5),
    dict(id="y_p_square", code="x = torch.randn(1024); x * x * x",
         ref=lambda x: [x * x * x], rtol=1e-6, atol=1e-6),
    dict(id="y_p_where", code="x = torch.randn(1024); y = torch.randn(1024); torch.where(x > 0, x, y)",
         ref=lambda x, y: [np.where(x > 0, x, y)], rtol=0, atol=0),
    dict(id="y_p_minimum", code="torch.minimum(torch.randn(1024), torch.randn(1024))",
         ref=lambda a, b: [np.minimum(a, b)], rtol=0, atol=0),
    dict(id="y_p_gelu", code="torch.nn.functional.gelu(torch.randn(1024))",
         ref=lambda x: [f32(0.5 * f64(x) * (1 + np.vectorize(__import__("math").erf)(f64(x) / np.sqrt(2.0))))],
         rtol=1e-3, atol=1e-4),

    # reductions with other block sizes and awkward lengths
    dict(id="y_r_t256", code="torch.randn(8192).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-3, atol=1e-3, knobs=T256),
    dict(id="y_r_t64", code="torch.randn(2048).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=T64),
    dict(id="y_r_t32", code="torch.randn(1024).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=T32),
    dict(id="y_r_short_row", code="torch.randn(256, 64).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="y_r_odd_row", code="torch.randn(32, 1000).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="y_r_row_t256", code="torch.randn(64, 2048).mean(-1)",
         ref=lambda x: [f32(f64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=T256),
    dict(id="y_r_amax_rows", code="torch.amax(torch.randn(64, 512), dim=-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="y_r_3d", code="torch.randn(4, 16, 256).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),

    # fused
    dict(id="y_f_softmax_short", code="torch.softmax(torch.randn(128, 64), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="y_f_softmax_long", code="torch.softmax(torch.randn(8, 4096), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="y_f_softmax_scaled", code="torch.softmax(torch.randn(32, 512) * 0.5, dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x) * np.float32(0.5)))], rtol=1e-3, atol=1e-5),
    dict(id="y_f_softmax_t256", code="torch.softmax(torch.randn(32, 1024), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5, knobs=T256),
    dict(id="y_f_rmsnorm_wide", code="x = torch.randn(16, 4096); w = torch.randn(4096); "
         "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6) * f64(w))],
         rtol=1e-3, atol=1e-4),
    dict(id="y_f_center_scale", code="x = torch.randn(32, 128); (x - x.mean(-1, keepdim=True)) * 2.0",
         ref=lambda x: [f32((f64(x) - f64(x).mean(-1, keepdims=True)) * 2.0)], rtol=1e-3, atol=1e-4),
    dict(id="y_f_sum_norm", code="x = torch.rand(32, 256) + 0.1; x / x.sum(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) / f64(x).sum(-1, keepdims=True))], rtol=1e-3, atol=1e-4),

    # matmul
    dict(id="y_m_odd", code="torch.randn(100, 50) @ torch.randn(50, 70)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="y_m_96", code="torch.randn(96, 96) @ torch.randn(96, 96)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="y_m_1024", code="torch.randn(1024, 256) @ torch.randn(256, 512)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="y_m_scaled", code="torch.randn(128, 128) @ torch.randn(128, 128) * 0.5",
         ref=lambda a, b: [f32(f64(a) @ f64(b) * 0.5)], rtol=1e-3, atol=1e-3),
    dict(id="y_m_small", code="torch.randn(16, 32) @ torch.randn(32, 16)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="y_m_wide", code="torch.randn(32, 512) @ torch.randn(512, 1024)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),

    # attention
    dict(id="y_a_16", code="q = torch.randn(16, 16); kt = torch.randn(16, 16); v = torch.randn(16, 16); "
         "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 4.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="y_a_256x64", code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
         "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="y_a_dv", code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 64); "
         "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 5.656854) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="y_a_wide", code="q = torch.randn(32, 128); kt = torch.randn(128, 32); v = torch.randn(32, 128); "
         "torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) / 11.313708) @ f64(v))], rtol=1e-3, atol=1e-4),
]

# ---- round 3: data types (f16, i32, i64, f64) and tensor cores -------------------------------
f16 = lambda a: a.astype(np.float16)

OPEN += [
    dict(id="z_h_add", code="torch.randn(1024, dtype=torch.float16) + torch.randn(1024, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) + f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="z_h_scale", code="torch.randn(2048, dtype=torch.float16) * 2.0 + 1.0",
         ref=lambda a: [f16(f64(a) * 2.0 + 1.0)], rtol=1e-2, atol=1e-2),
    dict(id="z_h_relu_mul", code="x = torch.randn(1024, dtype=torch.float16); y = torch.randn(1024, dtype=torch.float16); torch.relu(x) * y",
         ref=lambda x, y: [f16(np.maximum(f64(x), 0) * f64(y))], rtol=1e-3, atol=1e-3),
    dict(id="z_h_exp", code="torch.exp(torch.randn(1024, dtype=torch.float16))",
         ref=lambda a: [f16(np.exp(f64(a)))], rtol=1e-2, atol=1e-2),
    dict(id="z_h_sum", code="torch.randn(4096, dtype=torch.float16).sum()",
         ref=lambda x: [f16(f64(x).sum())], rtol=2e-2, atol=5e-2, knobs=REDUCE_KNOBS),
    dict(id="z_h_rowsum", code="torch.randn(64, 512, dtype=torch.float16).sum(-1)",
         ref=lambda x: [f16(f64(x).sum(-1))], rtol=2e-2, atol=5e-2, knobs=REDUCE_KNOBS),
    dict(id="z_h_softmax", code="torch.softmax(torch.randn(32, 256, dtype=torch.float16), dim=-1)",
         ref=lambda x: [f16(_softmax(f64(x)))], rtol=2e-2, atol=1e-3),
    dict(id="z_h_matmul", code="torch.randn(128, 128, dtype=torch.float16) @ torch.randn(128, 128, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=5e-2),
    dict(id="z_h_matmul_rect", code="torch.randn(256, 64, dtype=torch.float16) @ torch.randn(64, 128, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=5e-2),   # 64-way split-K: use tools/check_partial.py
    dict(id="z_h_mm_256", code="torch.randn(256, 256, dtype=torch.float16) @ torch.randn(256, 256, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=1e-1),
    dict(id="z_h_mm_wide", code="torch.randn(64, 256, dtype=torch.float16) @ torch.randn(256, 512, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=1e-1),
    dict(id="z_h_mm_odd", code="torch.randn(100, 50, dtype=torch.float16) @ torch.randn(50, 70, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=5e-2),
    dict(id="z_h_mm_small", code="torch.randn(32, 32, dtype=torch.float16) @ torch.randn(32, 32, dtype=torch.float16)",
         ref=lambda a, b: [f16(f64(a) @ f64(b))], rtol=2e-2, atol=5e-2),
    dict(id="z_h_mm_relu", code="torch.relu(torch.randn(128, 128, dtype=torch.float16) @ torch.randn(128, 128, dtype=torch.float16))",
         ref=lambda a, b: [f16(np.maximum(f64(a) @ f64(b), 0))], rtol=2e-2, atol=5e-2),
    dict(id="z_h_mm_bias", code="torch.randn(128, 128, dtype=torch.float16) @ torch.randn(128, 128, dtype=torch.float16) + torch.randn(128, dtype=torch.float16)",
         ref=lambda a, b, c: [f16(f64(a) @ f64(b) + f64(c))], rtol=2e-2, atol=1e-1),
    dict(id="z_i_add", code="torch.randint(-50, 50, (1024,), dtype=torch.int32) + torch.randint(-50, 50, (1024,), dtype=torch.int32)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="z_i_mul", code="torch.randint(-50, 50, (1024,), dtype=torch.int32) * torch.randint(-50, 50, (1024,), dtype=torch.int32)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="z_i_max", code="torch.maximum(torch.randint(-50, 50, (1024,), dtype=torch.int32), torch.randint(-50, 50, (1024,), dtype=torch.int32))",
         ref=lambda a, b: [np.maximum(a, b)], rtol=0, atol=0),
    dict(id="z_l_add", code="torch.randint(-50, 50, (1024,), dtype=torch.int64) + torch.randint(-50, 50, (1024,), dtype=torch.int64)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="z_d_add", code="torch.randn(1024, dtype=torch.float64) + torch.randn(1024, dtype=torch.float64)",
         ref=lambda a, b: [a + b], rtol=1e-6, atol=1e-6),   # submit.py compares as f32
    dict(id="z_d_fma", code="x = torch.randn(1024, dtype=torch.float64); x * x + 1.0",
         ref=lambda x: [x * x + 1.0], rtol=1e-6, atol=1e-6),
    dict(id="z_d_sum", code="torch.randn(4096, dtype=torch.float64).sum()",
         ref=lambda x: [x.sum()], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),   # emmy accumulates in f32
]

# ---- round 4: f16 attention on tensor cores --------------------------------------------------
OPEN += [
    dict(id="w_h_attn_64", code="q = torch.randn(64, 64, dtype=torch.float16); kt = torch.randn(64, 64, dtype=torch.float16); "
         "v = torch.randn(64, 64, dtype=torch.float16); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f16(_softmax((f64(q) @ f64(kt)) / 8.0) @ f64(v))], rtol=3e-2, atol=1e-2),
    dict(id="w_h_attn_128", code="q = torch.randn(128, 64, dtype=torch.float16); kt = torch.randn(64, 128, dtype=torch.float16); "
         "v = torch.randn(128, 64, dtype=torch.float16); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f16(_softmax((f64(q) @ f64(kt)) / 8.0) @ f64(v))], rtol=3e-2, atol=1e-2),
    dict(id="w_h_attn_32", code="q = torch.randn(32, 32, dtype=torch.float16); kt = torch.randn(32, 32, dtype=torch.float16); "
         "v = torch.randn(32, 32, dtype=torch.float16); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [f16(_softmax((f64(q) @ f64(kt)) / 5.656854) @ f64(v))], rtol=3e-2, atol=1e-2),
    dict(id="w_h_softmax_mm", code="x = torch.randn(64, 128, dtype=torch.float16); w = torch.randn(128, 64, dtype=torch.float16); "
         "torch.softmax(x @ w, dim=-1)",
         ref=lambda x, w: [f16(_softmax(f64(x) @ f64(w)))], rtol=3e-2, atol=1e-3),
]

# ---- round 5: other copy engines and tensor-core modes, selected with schedule knobs ---------
MM128 = "torch.randn(128, 128, dtype=torch.float16) @ torch.randn(128, 128, dtype=torch.float16)"
MM128_REF = lambda a, b: [f16(f64(a) @ f64(b))]

OPEN += [
    dict(id="k_h_mm_sync", code=MM128, ref=MM128_REF, rtol=2e-2, atol=5e-2, knobs="STAGE=d2/smem"),
    dict(id="k_h_mm_async2", code=MM128, ref=MM128_REF, rtol=2e-2, atol=5e-2, knobs="STAGE=d2/smem-async"),
    dict(id="k_h_mm_async4", code=MM128, ref=MM128_REF, rtol=2e-2, atol=5e-2, knobs="STAGE=d4/smem-async"),
    dict(id="k_h_mm_f16acc", code=MM128, ref=MM128_REF, rtol=3e-2, atol=2e-1, knobs="F16_MMA_F32_ACC=1"),
    dict(id="k_mm_sync", code="torch.randn(128, 128) @ torch.randn(128, 128)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3, knobs="STAGE=d2/smem"),
    dict(id="k_mm_async3", code="torch.randn(128, 128) @ torch.randn(128, 128)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3, knobs="STAGE=d3/smem-async"),
    dict(id="k_h_mm_unroll", code=MM128, ref=MM128_REF, rtol=2e-2, atol=5e-2, knobs="UNROLL=1"),
]
