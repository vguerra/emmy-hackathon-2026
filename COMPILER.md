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
| ran on the RTX 5090 and matched the reference | the 15 open problems, plus 127 probe problems in `problems_extra/` generated from my own torch expressions (other shapes, ops, block sizes, dtypes, tensor cores, copy-engine knobs, and odd extents that no tile or thread count divides) | 142 |
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
- **Tensor-core stores pair columns only when that is safe.** A lane's two adjacent C columns go out as one 2-wide
  store when the row stride is even and any column mask has a known even bound; otherwise each column is stored on
  its own with its own bounds check (a 2-wide store needs a 2-element-aligned address, and a mask between the pair
  would drop the last column). Odd-extent probes (`o_h_mm_33x17x65`) found this.
- **Large shared memory** switches to a single `extern .shared` array with 16-byte-aligned slices.

## Performance

The launch geometry is fixed by `meta.json`, so the compiler's levers are the instructions each thread runs and how
its memory accesses overlap. Measured with `tools/bench.py` (median of 100 launches after warmup), before any of
this against after all of it, on the problems where it mattered:

| problem | before (µs) | after (µs) | speed-up |
|---|---|---|---|
| m01_matmul_512 (f32, shared-memory tiles) | 99.0 | 21.2 | 4.7x |
| y_m_1024 (f32 matmul 1024x256x512) | 92.9 | 19.1 | 4.9x |
| x_a_noscale (f32 attention, 1 072 µs) | 1071.6 | 164.4 | 6.5x |
| y_a_256x64 (f32 attention) | 3870.0 | 537.2 | 7.2x |
| w_h_softmax_mm (f16 matmul + softmax) | 3107.3 | 1311.1 | 2.4x |
| a01_attention | 115.3 | 53.9 | 2.1x |
| y_a_wide (f32 attention) | 240.3 | 70.1 | 3.4x |

Most problems run in 3-5 µs warm and did not change. What was done:

1. **Index arithmetic is value-numbered** (`int_op`): immediates fold, `x+0` / `x*1` vanish, and a repeated
   `(op, operand, operand)` is emitted once per straight-line region (any label forgets). The inner loop of the
   f32 matmul spent ~60 integer instructions on every 4 multiply-adds, most of it recomputed `div`/`rem`/`mul`
   chains. This alone took `m01` from 99 to 40 µs.
2. **Innermost loops with a static trip count are unrolled by 4**, so their independent loads sit in one basic
   block and `ptxas` can issue them together instead of waiting out each load's latency inside a two-instruction
   loop body. Factor 4 was chosen by measuring 2, 4, 8 and 16.
3. **Multiply-add may fuse when the tolerance is non-zero.** With a zero tolerance every arithmetic op keeps its
   `.rn` spelling, which forbids `ptxas` from contracting it. A *rescaled* accumulator (`Accum.base`, the online-softmax
   `acc = acc*alpha + ...`) always stays strict: fusing the multiply into it made the small attention kernels
   `a02_attention_128` and `y_a_dv` about 10% slower than the baseline, while plain accumulators and plain
   assignments gain from fusing. Found by timing each kernel with unrolling and fusion switched off in turn.
4. **Consecutive loads of an unrolled loop become one vector load.** In the f32 attention kernels the query row
   `q[row, k]` is read one element per iteration. When four unrolled copies read four consecutive elements, the
   first copy issues one `ld.global.v4.f32` and the other three reuse its registers. Only done when it is provably
   safe: the load is a direct child of the loop body, its last index is exactly the loop axis, the loop starts at a
   multiple of 4 with step 1, the buffer's row length is a multiple of 4 (so every group is 16-byte aligned), the
   elements are 2 or 4 bytes, and the body never writes that buffer. Buffer base addresses are assumed 16-byte
   aligned, as `cudaMalloc` guarantees. On top of the other three it made the larger f32 attention kernels another 1.6x-3.3x faster.

Tried and dropped because they were slower or measured as no change:

- **Merging sibling loops** (loop jamming) so a thread's ten row-accumulators advance side by side and share
  loads. It was correct but 1.1-2.4x *slower* at every merge count and unroll factor tried. The cause is `ptxas`:
  on the original separate loops it unrolls and pipelines them itself (128 registers, 35 KB for `a01_attention`),
  and on a larger merged body it does not (48 registers, 17 KB). The compiler's structure feeds `ptxas`'s own
  heuristics, so a bigger loop body is not automatically better.
- **A different unroll factor per kernel.** Factors 8, 16 and 32 beat 4 on a couple of kernels
  (`y_a_256x64` -20% at 16) and lost on others (`a01_attention` +15% at 8), so one global factor stays.
- **Vector loads from shared memory** in the f32 matmul (`_a_smem[row, _ki]` is contiguous in the unrolled axis, so
  the same grouping applies). Correct, 142/142 pass and the inner loop issues `ld.shared.v4`, but `m01_matmul_512`
  (21.2 µs) and `y_m_1024` (19.1 µs) did not move: the matmul is bound by the global-to-shared pipeline and launch
  cost, not by shared-load issue slots. Removed rather than carry code with no measured gain.
- `ld.global.nc` for read-only inputs (the plain load already uses the L1 cache on these GPUs).

**Bandwidth.** `python tools/bandwidth.py <folder>` (problems from `tools/expressions_bandwidth.py`, 64 MiB buffers,
past the L2) reports achieved GB/s: elementwise `a + b` 1676 GB/s (94% of the 5090's ~1792), `x*2+1` 1740 (97%),
row sum 1591 (89%). A 16.8M-element full sum reaches 837 (47%): the IR fixes its grid at 64 blocks of 128 threads.

**Full comparison.** [`docs/perf-comparison.md`](docs/perf-comparison.md) times every single-kernel problem (112) on the
baseline commit `2fca374` and on the optimised compiler: 17 faster, 94 unchanged, 1 flagged slower (`f02_row_softmax`,
3.0 to 3.6 µs, a launch-bound kernel, so timer noise), geometric mean 1.13x. The unchanged ones are launch-bound at
3-5 µs. Regenerate it with `python tools/compare_commits.py 2fca374 HEAD`.

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
- `tests/test_ptx_structure.py` — compiles all 150 kernels and lints the PTX without `ptxas`: ASCII only, parameters
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
- A cooperative Tile whose `N` is not a multiple of `block_threads`. None of the odd-length reductions, thread counts
  (`t32`..`t256`), default schedules or grid splits in the `o_` probes makes emmy emit one: its reductions always give
  each block a whole number of threads.
