# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Emmy Hackathon 2026: a **compiler from Emmy Kernel IR to PTX**. The submission contract is fixed:

```bash
python compile.py path/to/<kernel>.ir.json > out.ptx   # PTX only on stdout; debug output goes to stderr
```

It is run mechanically against a hidden set of IRs in five categories (pointwise, reductions, matmul, fused,
attention-like); the organisers say the problem set is a baseline filter and submissions are then reviewed by hand,
with extra credit for tensor cores, other copy engines and data types, and for tests, examples and a clear feature
list. `COMPILER.md` is that feature list; keep it accurate when behaviour changes. Deadline: 23:59 AoE Tue Oct 27
(public GitHub repo plus a 3-5 minute video).

`TASK.md` holds the full IR statement/expression reference (nodes marked `*` are absent from the open and hidden
problems). **Hard rule:** the IR to PTX translation must be our own. Emmy may only load the IR (`Graph.from_dict`);
never lower through its renderer, `emmy compile` or `nvcc`.

## Commands

```bash
.venv/bin/python -m unittest discover -s tests -t .      # offline unit tests + PTX lint of every problem
python tools/run_all.py [dir ...] [-k substr]            # compile + run + compare on the hosted runner (network)
python tools/check_assembles.py                          # ptxas-only check of problems_asm/ (TMA)
python tools/make_examples.py [--check]                  # regenerate / verify examples/compiled/
python tools/bench.py <problem ...> [--json f | --compare f]   # warm median timing; use it before keeping an optimisation
python submit.py <problem-or-folder> --compiler compile.py
python tools/generate.py tools/expressions_extra.py --emmy .venv/bin/emmy --out problems_extra --only <id>
```

Python 3.12 with `emmy-ml==0.3.11`, numpy, requests, and torch (torch only to *generate* probe IRs). There is no pytest
and no pip in `.venv`; tests use `unittest`. macOS: no `timeout`, BSD `sed -i` needs a suffix argument.

## Layout

- `compile.py` — CLI; `compile_ir()` runs load -> prologue -> walk -> render.
- `ptxgen/` — the lowering, one module per concern (see `COMPILER.md` for the list). Annotate function signatures
  only, not local variables.
- `problems/` — the 15 organiser problems. `problems_extra/` — 104 probes made from `tools/expressions_extra.py`
  (prefixes: `x_`/`y_` shapes and ops, `z_` dtypes and tensor cores, `w_` f16 attention, `k_` schedule knobs).
  `problems_asm/` — assemble-only kernels. `examples/compiled/` — generated PTX beside its IR.

## Contracts and gotchas

- `meta.json`, not the IR node, defines the signature: `arg_order` gives parameter order and `args` gives each
  buffer's dtype and shape (loads carry no dtype). The entry name is the IR kernel `name`.
- The `gid < N` guard is not in the IR; it comes from the Tile. A cooperative Tile must have `N` a multiple of its
  block size (barriers sit below the guard).
- PTX: starts with `.version`/`.target`/`.address_size`, ASCII only. `.target sm_89` / `.version 8.0`; TMA kernels
  raise the target to `sm_90`. Arithmetic is spelled `.rn` so `ptxas` cannot fuse mul+add (several tolerances are 0).
- f16 values live in f32 registers and are rounded after every f16-typed op; see `ptxgen/dtypes.py`.
- `submit.py` compares outputs after casting to float32, so an f64 problem cannot have tolerance 0, and it sends
  intermediate buffers inline, which the runner refuses past ~500k values (`tools/run_all.py` routes those through
  `tools/check_partial.py`).
- Emmy accumulates reductions in f32 even for f64 inputs (the IR says so), and cannot trace `aten.max`/`amin`,
  `log_softmax`, or layernorm as one kernel. Schedule knobs (`EMMY_KNOBS`, e.g. `STAGE=d2/smem-async`,
  `WORK=t256,REDUCE=coop`) select other IR shapes; `STAGE=...smem-tma` needs `--target sm_120` and cannot be packaged
  by `generate.py`.
- Performance: `submit.py` times one cold launch (launch overhead dominates); use `tools/bench.py` for warm timings.
- Unsupported constructs raise `NotImplementedError` naming the construct; never emit guessed PTX.
