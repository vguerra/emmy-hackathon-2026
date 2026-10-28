# Emmy Kernel IR to PTX compiler

A compiler from Emmy's Kernel IR straight to PTX, written from scratch for the Emmy Hackathon 2026.

```bash
python compile.py path/to/kernel.ir.json > out.ptx      # PTX on stdout, nothing else
```

The translation is entirely this repository's own: Emmy is used only to *load* the IR (`Graph.from_dict`). Nothing
here calls Emmy's renderer, `emmy compile` or `nvcc`.

**At a glance**

- Passes all 15 organiser problems and 127 probe problems I generated (142 of 142 on the hosted RTX 5090).
- Tensor cores (`mma.sync` + `ldmatrix`, fused f16 flash attention), `cp.async` pipelines, swizzles, TMA.
- `f16`, `f32`, `f64`, `i32`, `i64`.
- Up to 7x faster than my first version on attention kernels, and 94-97% of peak DRAM bandwidth on large
  elementwise kernels.
- 53 offline unit tests plus a PTX lint of every kernel. Unsupported constructs raise a `NotImplementedError` that
  names them; the compiler never emits guessed PTX.

The full feature list is in [COMPILER.md](COMPILER.md).

## Try it

```bash
python -m venv .venv && .venv/bin/pip install emmy-ml==0.3.11 numpy requests    # Python 3.12; torch is only for generating new IRs
.venv/bin/python compile.py problems/m01_matmul_512/*.ir.json | head -40
.venv/bin/python submit.py problems/a01_attention --compiler compile.py          # compile, run on the hosted GPU, compare
.venv/bin/python -m unittest discover -s tests -t .                               # offline: tests + PTX lint of all 150 kernels
.venv/bin/python tools/run_all.py                                                 # every problem on the runner: 142/142
```

`submit.py` needs network access to the hosted runner (see [docs/HACKATHON.md](docs/HACKATHON.md) for the task as the
organisers stated it). Compiled examples, with their IR beside them, are in [examples/compiled/](examples/compiled/).

## Architecture

```
compile.py          CLI and pipeline: load -> prologue -> walk -> render
ptxgen/
  loader.py         IR + meta.json, kernel and Tile lookup, argument order and types
  emitter.py        PtxEmitter: lines, virtual registers, symbol tables, final text assembly
  dtypes.py         element types, register classes, immediates, conversions
  prologue.py       parameters, thread id, bounds guard, axis decode
  expr.py           index and predicate expressions, value numbering of integer math
  memory.py         addressing, Load/Write, cp.async, Smem, swizzle, vector-load grouping
  elementwise.py    Assign dispatch by operand type; transcendental helpers
  control.py        loops (with unrolling), Cond, Accum, shuffles, TreeHalve, statement dispatcher
  tensorcore.py     mma.sync, ldmatrix, fragment statements, epilogue stores
  tma.py            TMA copies and mbarriers
```

The IR is a tree of statements; each statement has an `emit_*` function and the dispatcher in `control.py` walks the
tree once. PTX registers are virtual (`ptxas` allocates), so there is no register allocator to get wrong.

## What is different here

- **Fusion is decided per accumulator, not per kernel.** With a non-zero tolerance in `meta.json`, multiply-add may
  fuse (`fma`); with tolerance zero everything is spelled `.rn` so `ptxas` cannot contract it. A *rescaled*
  accumulator (the online-softmax `acc = acc*alpha + ...`) stays strict even when fusing is allowed: fusing it made
  two small attention kernels ~10% slower, found by timing each kernel with each optimisation switched off in turn.
- **Vector loads with a stated safety proof.** Four unrolled copies reading four consecutive elements become one
  `ld.global.v4`, but only when the load is a direct child of the loop body, its last index is exactly the loop
  axis, the start is a multiple of 4 with step 1, the row length is a multiple of 4, and the body never writes the
  buffer ([`grouped_loads`](ptxgen/memory.py)). Each condition has a test. It took the larger f32 attention kernels
  another 1.6x-3.3x.
- **f16 without a second set of operations.** f16 values live in f32 registers and are rounded after every f16-typed
  op, which is how CUDA evaluates `__half` arithmetic; every elementwise op therefore supports f16 for free.
- **Fragment registers cannot be indexed at run time**, so loops that name a fragment by its axis are unrolled at
  compile time with the axis bound to an immediate.
- **Swizzle as one address transform** applied to both the `cp.async` fill and the `ldmatrix` drain, so the two agree
  by construction (and the transform is unit-tested against a reference).
- **Probes that find bugs.** Odd-extent probes found a misaligned f16 store (33x17x65 matmul crashed with
  `cudaErrorMisalignedAddress`) and a dropped last column; the fix and its tests are in
  `ptxgen/tensorcore.py` and `tests/test_lowering.py`.
- **Negative results are recorded.** Merging loops ("jamming") was correct but 1.1-2.4x slower because `ptxas`
  unrolls the separate loops better itself; vector loads from shared memory did not move the f32 matmul; per-kernel
  unroll factors were noise. See "Tried and dropped" in [COMPILER.md](COMPILER.md).

## Performance

Warm median of 100 launches on the runner (`tools/bench.py`). The organiser's `submit.py` times one cold launch, which
is mostly launch overhead for small kernels, so it is not used for the numbers below.

| problem | first version (us) | now (us) | speed-up |
|---|---:|---:|---:|
| y_a_256x64 (f32 attention) | 3870.0 | 537.2 | 7.2x |
| x_a_noscale (f32 attention) | 1071.6 | 164.4 | 6.5x |
| y_m_1024 (f32 matmul 1024x256x512) | 92.9 | 19.1 | 4.9x |
| m01_matmul_512 | 99.0 | 21.2 | 4.7x |
| y_a_wide (f32 attention) | 240.3 | 70.1 | 3.4x |
| w_h_softmax_mm (f16) | 3107.3 | 1311.1 | 2.4x |
| a01_attention | 115.3 | 53.9 | 2.1x |

Most problems run in 3-5 us, which is the launch floor, and did not change. Regenerate the whole comparison with
`python tools/compare_commits.py 2fca374 HEAD`.

**Achieved bandwidth** on large memory-bound kernels (`tools/bandwidth.py`, RTX 5090, peak about 1792 GB/s, buffers
past the 96 MiB L2 so the numbers are DRAM):

| problem | bytes moved | time (us) | GB/s | of peak |
|---|---:|---:|---:|---:|
| `a + b`, 16.8M f32 | 201 MB | 120.1 | 1676 | 94% |
| `x*2+1`, 16.8M f32 | 134 MB | 77.2 | 1740 | 97% |
| row sum, 8192x4096 f32 | 134 MB | 84.4 | 1591 | 89% |
| full sum, 16.8M f32 | 67 MB | 80.2 | 837 | 47% |

The full sum is limited by the launch geometry the IR fixes (64 blocks of 128 threads), not by the instruction stream;
the compiler cannot change the grid.

## Verification

| level | what | count |
|---|---|---|
| ran on the RTX 5090 and matched the reference | 15 organiser problems + 127 probes in `problems_extra/` | 142 |
| assembled with `ptxas`, not executed | a TMA matmul in `problems_asm/` (the runner API cannot pass a tensor-map argument) | 1 |

Offline: `tests/` has 53 unit tests (immediates, conversions, index expressions, swizzle executed against a
reference, load grouping, tensor-core stores) and lints the PTX of all 150 kernels without `ptxas`.

## Not supported

`bf16`, fp8 and fp4 (the runner API has no such buffers, so they could not be checked), `wgmma` and `SetMaxNReg`
(Hopper `sm_90a` only; the runner is `sm_120`), f16-accumulate `mma`, `Tile.aux_threads`, warp shuffles of 64-bit
values, a cooperative Tile whose `N` is not a multiple of its block size. Each raises a `NotImplementedError` naming
the construct. TMA is complete for the statements Emmy emits but only its PTX syntax is confirmed, not the data
movement.

## Layout

| path | contents |
|---|---|
| `compile.py`, `ptxgen/` | the compiler |
| `problems/` | the 15 organiser problems (IR, `meta.json`, reference) |
| `problems_extra/` | 127 probes made from `tools/expressions_extra.py` (`x_`/`y_` shapes and ops, `z_` dtypes and tensor cores, `w_` f16 attention, `k_` schedule knobs, `o_` odd extents) |
| `problems_asm/` | assemble-only kernels (TMA) |
| `examples/compiled/` | generated PTX beside its IR |
| `tests/` | offline unit tests and PTX lint |
| `tools/` | `run_all.py`, `bench.py`, `bandwidth.py`, `compare_commits.py`, `generate.py`, `make_examples.py`, `check_assembles.py` |
| `docs/` | `IR-REFERENCE.md`, `perf-comparison.md`, the organisers' README and API notes |
