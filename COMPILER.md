# The compiler

`compile.py` lowers one Emmy Kernel IR file to PTX. The lowering is this repository's own: emmy is used only to
*load* the IR (`Graph.from_dict`), never to render it.

```bash
python compile.py path/to/<kernel>.ir.json > out.ptx        # PTX on stdout, nothing else
python compile.py path/to/<kernel>.ir.json --dump-graph     # also pretty-prints the IR to stderr
```

## What it supports

| area | supported |
|---|---|
| **statements** | `Tile`, `Load`/`Write` (scalar, `v2`/`v4`, atomic), `Assign`, `Let`, `Select`, `Init`, `Accum` (including `base`), `Loop`, `StridedLoop`, `Cond`, `Smem`, `Sync` (block, warp, named), `WarpShuffle`, `TreeHalve` |
| **data types** | `f32`, `f16`, `f64`, `i32`, `i64`, with per-value type tracking and conversions between them |
| **elementwise ops** | `add subtract multiply divide maximum minimum sqrt reciprocal rsqrt abs negative square copy relu exp exp2 log log2 sigmoid tanh silu softplus erf gelu`, comparisons, `where` |
| **tensor cores** | `mma.sync` m16n8k16 (f16 in, f32 accumulate): `RegFragment`, `LdmatrixLoad` (staged `x4` / `x4.trans` pair / `x2`, and gathered straight from global memory with K and row masks), `MmaSyncPtx`, `RegStore` (with fused epilogue, guards, atomic), `FragmentApply`, `FragmentRowReduce`, `FragmentRepack` — i.e. fused flash attention on tensor cores |
| **copy engines** | register-staged copy (`ld`/`st`), `cp.async` 4/8/16 bytes with commit/wait groups at any pipeline depth, software smem swizzles B32/B64/B128, TMA bulk-tensor copies with mbarriers |
| **shared memory** | static `.shared`, or one `extern` dynamic array once the kernel needs more than 48 KiB |

### How it was verified

| level | what | count |
|---|---|---|
| ran on the RTX 5090 and matched the reference | the 15 open problems, plus 104 probe problems in `problems_extra/` generated from my own torch expressions (other shapes, ops, block sizes, dtypes, tensor cores, copy-engine knobs) | 119 |
| assembled with `ptxas`, not executed | a TMA matmul in `problems_asm/` — the runner API cannot pass a tensor-map argument | 1 |

TMA cannot be run through this evaluation at all: the problem generator itself fails to package a TMA kernel
(its argument list contains the descriptor, which is not a graph buffer). The lowering is complete for the
statements emmy emits (`TmaDescriptor`, `TmaLoad`, `MbarrierInit`, `MbarrierArriveExpectTx`, `MbarrierArrive`,
`MbarrierWait`) but only the PTX syntax is confirmed, not the data movement.

## Design

```
compile.py          CLI and the pipeline: load -> prologue -> walk -> render
ptxgen/
  loader.py         IR + meta.json, kernel and Tile lookup, argument order and types
  emitter.py        PtxEmitter (lines, virtual registers, symbol tables), final text assembly
  dtypes.py         element types, register classes, immediates, conversions
  prologue.py       parameters, thread id, bounds guard, axis decode
  expr.py           index and predicate expressions
  memory.py         addressing, Load/Write, cp.async, Smem, swizzle
  elementwise.py    Assign dispatch by operand type; transcendental helpers
  control.py        loops, Cond, Accum, shuffles, TreeHalve, and the statement dispatcher
  tensorcore.py     mma.sync, ldmatrix, fragment statements, RegStore
  tma.py            TMA copies and mbarriers
```

Decisions worth knowing about:

- **`meta.json` is the contract.** Parameter order comes from `arg_order`; element types and shapes come from the
  meta `args`, because loads carry no dtype of their own.
- **Every SSA value has a register class** (`f`, `d`, `r`, `rd`, `p`). f16 values live in f32 registers and are
  rounded back after each operation that yields an f16 — how CUDA evaluates `__half` arithmetic — and widened/narrowed
  at loads and stores. This one rule gives f16 to every elementwise op without duplicating them.
- **Arithmetic is spelled `.rn`** (`mul.rn.f32`) so `ptxas` cannot fuse a multiply and an add into an fma; the
  reference outputs round every operation, and several problems have tolerance 0.
- **The bounds guard is derived, not read.** It is not in the IR; it is `gid < N` from the Tile. A cooperative Tile
  must have `N` as a multiple of its block size, because shuffles and barriers sit below the guard.
- **Registers are virtual.** `ptxas` allocates; the compiler declares one more than it handed out.
- **Fragment registers cannot be indexed at run time**, so a loop whose body names a fragment by its axis
  (`_c0_[_r0]`) is unrolled at compile time with the axis bound to an immediate.
- **Swizzle is an address transform**, `e ^ (((e >> shift) & mask) << 3)`, applied to the fill (`cp.async`) and the
  drain (`ldmatrix`) alike, so they agree by construction.
- **Large shared memory** switches to a single `extern .shared` array with 16-byte-aligned slices.

## Testing

```bash
python -m unittest discover -s tests -t .            # offline: unit tests + PTX lint of every problem (no GPU, no network)
EMMY_RUNNER_TESTS=1 python -m unittest discover -s tests -t .   # also runs the hosted runner
python tools/run_all.py                              # every problem on the runner, one PASS/FAIL line each
python tools/run_all.py problems_extra -k f16        # a subset
python tools/check_assembles.py                      # the assemble-only (TMA) problems
python tools/make_examples.py --check                # examples/compiled is current
python tools/bench.py problems_extra/y_m_1024        # warm timing; --json / --compare to diff two builds
```

- `tests/test_dtypes.py`, `tests/test_lowering.py` — unit tests of immediates, conversions, index expressions,
  swizzle (executed against a reference), arithmetic dispatch, rendering.
- `tests/test_ptx_structure.py` — compiles all 127 kernels and lints the PTX without `ptxas`: ASCII only, parameters
  equal `arg_order`, no register used past its declaration, no branch to a missing label.
- `tools/expressions_extra.py` — the torch expressions behind `problems_extra/`; regenerate one with
  `python tools/generate.py tools/expressions_extra.py --emmy <emmy> --out problems_extra --only <id>`.
- `tools/check_partial.py` — for split-K problems whose partial buffer is too large for `submit.py` to chain
  (it sends intermediates inline as JSON); runs kernel 1 and reduces the partials locally.

`examples/compiled/` holds the compiler's output beside the IR for seven kernels, from pointwise to f16 attention.

## Not supported

Each of these stops with a `NotImplementedError` naming the construct, never with wrong PTX.

- `bf16`, fp8 and fp4 types, and Volta `m8n8k4` fragments. The runner's API has no `bf16`/fp8 buffers, so they
  could not be checked.
- `wgmma` and `SetMaxNReg` (Hopper `sm_90a` only; the runner is an `sm_120` part).
- f16-accumulate `mma` (packed C fragments, `FragmentPromote`), `Tile.aux_threads` (warp-specialised producers),
  `TreeHalve.inner`, warp shuffles of 64-bit values, transcendental ops on `f64`.
- A cooperative Tile whose `N` is not a multiple of `block_threads`.
