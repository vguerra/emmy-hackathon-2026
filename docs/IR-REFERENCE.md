# Kernel IR reference

A guide to every Emmy Kernel IR operation, with how this compiler lowers it. The authoritative argument lists are in
[`TASK.md`](../TASK.md); the compiler's overall feature list is [`COMPILER.md`](../COMPILER.md).

Operations marked `*` appear in **no** open or hidden problem (that is the organisers' marking). The last column says
what this compiler does with each one: **yes** is implemented, **no** raises `NotImplementedError` naming the
construct, and never emits guessed PTX.

## Reading a kernel

A kernel is a tree of statements rooted at a `Tile`. Statements bind SSA names (`acc = add(acc, v)`); indices,
predicates and loop bounds are *expressions*, not statements. Three facts explain most surprises:

- Loads carry no dtype. Element types and shapes come from `meta.json` (`args`), and parameter order from `arg_order`.
- The bounds guard `gid < N` is not in the IR. It is derived from the Tile's `N`.
- A cooperative Tile (one that uses barriers or shuffles) must have `N` a multiple of its block size, because those
  sit below the guard.

```
Tile[a0_b, a0_u] (N=1024)          <- the block: 1024 cells of work
    x = load in0[a0_b*256 + a0_u]      <- Load
    y = relu(x)                        <- Assign
    out[a0_b*256 + a0_u] = y           <- Write
```

## Program and launch

| op | arguments | what it does | here |
|---|---|---|---|
| `KernelOp` | `name`, `body`, `zero_delegated`, `serial` | One GPU kernel. Its parameters are derived from the body. `zero_delegated` lists outputs an earlier kernel zeroes; `serial` lists axes launched one coordinate at a time (empty in every problem). | yes (the entry name is the kernel name) |
| `Tile` | `axes`, `body`, `block_threads`, `aux_threads`, `raster_axes`, `raster_group`, `raster_orient` | Maps the iteration space onto the thread grid, decodes each thread's axis indices and runs the body once per cell. | yes, except `aux_threads` (warp-specialised producers) |
| `Smem` | `name`, `extents`, `dtype`, `align` | Declares one shared array for the block. | yes: static `.shared`, or a slice of one `extern` array past 48 KiB, 16-byte aligned |
| `Sync` | `barrier_id`, `count`, `warp` | A barrier: block-wide, warp-wide, or a named barrier over `count` threads. | yes: `bar.sync`, `bar.warp.sync`, `bar.sync id, count` |

## Scalar compute

| op | arguments | what it does | here |
|---|---|---|---|
| `Load` | `names`, `input`, `index`, `dtype` | Reads one value, or several consecutive ones, from a buffer into SSA names. | yes; several names become `ld.v2`/`ld.v4`. Consecutive loads across unrolled loop copies are merged when provably safe |
| `Write` | `output`, `index`, `values`, `value_dtype`, `atomic`, `swizzle` | Stores one value or a group, optionally as an atomic add. | yes: `st` / `st.v2` / `st.v4`, `red.global.add` for atomics |
| `Assign` | `name`, `op`, `args`, `dtype` | `name = op(args)`: one elementwise operation. | yes, the op list is in `COMPILER.md` (arithmetic, `exp`, `log`, `tanh`, `erf`, `gelu`, comparisons, `where`, ...) |
| `Let` | `name`, `value`, `dtype` | Binds one pure expression to a name: a literal, an index, or a flat offset. | yes |
| `Init` | `name`, `identity`, `dtype` | Declares carried state seeded with a literal, so a masked tail can select the identity or a chunk loop can update it in place. | yes |
| `Accum` | `name`, `value`, `op`, `dtype`, `axes`, `base` | Folds `value` into an accumulator that starts at the op's identity and stays visible after the loop. `base` marks a rescaled accumulator (`acc*alpha + ...`, online softmax). | yes; a rescaled accumulator is never fused into an fma |
| `Select`* | `name`, `branches` | Binds `name` to the branch whose coordinate predicate holds. | yes |
| `Pre`* | `name`, `carrier`, `index` | Reads one cell of carried state: the previous step's value, or the final value after the loop. | no |
| `Carry`* | `name`, `value`, `index`, `seed`, `dtype` | Sets a carried-state cell's next value; steps run in order. | no |
| `ZeroPrologue`* | `dst`, `words` | Zeroes another kernel's atomic accumulator from an earlier kernel, replacing a memset. | no |

## Control flow

| op | arguments | what it does | here |
|---|---|---|---|
| `Loop` | `axis`, `body`, `unroll`, `seed` | Runs the body once per value of `axis`, folding any accumulators inside. | yes; a static innermost loop is unrolled by 4 |
| `StridedLoop` | `axis`, `start`, `step`, `body`, `unroll`, `end`, `seed` | Loops from `start` in steps of `step`, the form cooperative threads use to stride an axis. | yes |
| `Cond` | `cond`, `body`, `else_body` | Runs `body` when the predicate holds, otherwise `else_body`. | yes (predicated branch) |

## Copies into shared memory

| op | arguments | what it does | here |
|---|---|---|---|
| `CpAsyncCopy` | `smem`, `smem_index`, `src`, `src_index`, `nbytes`, `swizzle`, `lane_index`, `lane_rows` | Copies 4, 8 or 16 bytes from global memory straight into shared memory with one `cp.async`. | yes; 16-byte copies use `.cg`; swizzles B32/B64/B128 |
| `CpAsyncCommit` | none | Closes the thread's preceding copies into one commit group. | yes |
| `CpAsyncWait` | `group` | Blocks until at most `group` groups are still in flight. | yes; pipelines of any depth |
| `TmaDescriptor`* | `name`, `src_buf`, `src_shape`, `box_extents`, `swizzle`, `dtype` | Declares the host-built tensor map a TMA copy reads; renders nothing in the kernel. | yes |
| `TmaLoad`* | `smem`, `smem_index`, `desc`, `coords`, `mbar`, `mbar_slot` | Copies one box into shared memory with a single-thread `cp.async.bulk.tensor`. | yes (assembled, not executed) |
| `MbarrierInit`* | `mbar`, `count`, `slot` | Initialises one mbarrier, or one slot of an array, in the prologue. | yes |
| `MbarrierArriveExpectTx`* | `mbar`, `bytes_`, `slot` | Announces the byte count of the TMA copy about to arrive. | yes |
| `MbarrierArrive`* | `mbar`, `slot` | Signals arrival without a byte count (a consumer marking a slot empty). | yes |
| `MbarrierWait`* | `mbar`, `phase`, `slot` | Blocks until the mbarrier's parity flips for `phase`. | yes |
| `SetMaxNReg`* | `count`, `direction` | Shrinks or grows a warp's register budget (`sm_90` and newer). | no |

The TMA statements raise the PTX target to `sm_90`. The runner API cannot pass a tensor-map argument, so only the PTX
syntax is confirmed (`problems_asm/`, `tools/check_assembles.py`), not the data movement.

## Combining across threads

| op | arguments | what it does | here |
|---|---|---|---|
| `WarpShuffle` | `state`, `state_b`, `combine_states`, `length`, `dtype` | Folds per-lane partial states over `length` lanes with a register-only XOR shuffle butterfly. | yes (`shfl.sync.bfly`); not 64-bit values |
| `TreeHalve` | `bufs`, `state`, `state_b`, `combine_states`, `length`, `tid_var`, `dtype`, `barrier_id`, `barrier_count`, `inner` | Reduces a power-of-two shared-memory tree of partial states and broadcasts the result, for combines wider than a warp. | yes, except `inner` |

## Tensor cores: register fragments (`mma.sync`)

The compiler implements the f16-in, f32-accumulate `m16n8k16` path, which is enough for fused flash attention.

| op | arguments | what it does | here |
|---|---|---|---|
| `RegFragment`* | `name`, `role`, `shape`, `dtype`, `count`, `nregs` | Declares a per-thread register array for the A, B or C operand; C starts at zero. | yes |
| `LdmatrixLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm`, `swizzle`, `staged`, `gmem_guard`, `k_zero`, `b_trans`, `pair_frag`, ... | Loads an operand fragment into registers from shared memory, or straight from global memory with guards. | yes: staged `x4`, `x4.trans` pairs, `x2`, and gathered global loads with K and row masks; no byte-slab dequantisation |
| `MmaSyncPtx`* | `c_frag`, `a_frag`, `b_frag`, `shape`, `ab_dtype`, `c_dtype`, `b_row_major`, ... | Issues one `mma.sync`: `c = a·b + c`. | yes (f16 in, f32 accumulate) |
| `FragmentApply`* | `out`, `op`, `args`, `kinds`, `in_place`, `layout`, `post`, `row_base`, `col_base` | Applies one elementwise op to every element of a C fragment. An argument is a fragment, a per-row pair, a uniform scalar, a coordinate mask, or a global load (a bias). | yes |
| `FragmentRowReduce`* | `top`, `bot`, `frags`, `op`, `layout`, `dtype` | Reduces a warp's C fragments per row (the softmax max or sum). | yes |
| `FragmentRepack`* | `frag`, `srcs`, `ab_dtype`, `fragment_layout`, `part` | Converts C fragments into one 16-bit A fragment, so a result can feed the next matmul (the flash-attention step). | yes |
| `RegStore`* | `dst_buffer`, `dst_index`, `frag`, `shape`, `ldm`, `epilogue`, `extra_frags`, `m_guard`, `n_guard`, `atomic`, ... | Stores a C fragment per lane after an optional fused pointwise epilogue and a downconvert. | yes: guards, atomic, fused bias/activation epilogue; paired stores only when alignment is provable |
| `FragmentPromote`* | `dst`, `src` | Adds a packed f16 accumulator into its f32 shadow fragment. | no (f16-accumulate `mma`) |
| `BlockScaleLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm` | Loads a lane's block scales for the block-scaled fp4 `mma`. | no |

`RegStore.epilogue` is a pure `Lambda`* (`params`, `body`, `results`): the projection tail's own `Load`, `Assign` and
`Select` statements, evaluated per fragment element. `FragLayout`* is the per-atom geometry (`n_elems`, `elem_row`,
`reduce_xors`, `lane_decl`, `row_off`, `col_off`) the fragment operations read.

## Tensor cores: warp groups (`wgmma`)

`WgmmaDescriptor`*, `WgmmaMma`*, `WgmmaFence`*, `WgmmaCommit`* and `WgmmaWait`* build a shared-memory matrix
descriptor, issue an asynchronous `wgmma.mma_async` across the four warps of a group, order earlier accesses, close a
group of cells, and wait for groups in flight. They need Hopper `sm_90a`; the hosted runner is an `sm_120` part, so
none of them is supported.

## Expressions

| expr | arguments | meaning | here |
|---|---|---|---|
| `Var` | `name` | An axis variable or SSA name. | yes |
| `Literal` | `value`, `dtype` | A numeric constant. | yes |
| `BinaryExpr` | `op`, `left`, `right` | Arithmetic, comparison or logic on two expressions. | yes; integer math is value-numbered |
| `TernaryExpr` | `cond`, `if_true`, `if_false` | Chooses between two expressions by a predicate. | yes |
| `Builtin`* | `name` | A GPU built-in such as `threadIdx.x`. | no |
| `FuncCallExpr`* | `name`, `args` | A call to a registered elementwise function. | no |
| `CastExpr`* | `dtype`, `expr` | Converts an expression to another scalar type. | no |
| `FlatIndex`* | `buffer`, `index` | The row-major element offset of a coordinate, flattened against the buffer's shape. | no |

## Where each category uses what

| category | statements |
|---|---|
| pointwise | `Tile`, `Load`, `Assign`, `Write` |
| reductions | plus `StridedLoop`, `Accum`, `WarpShuffle`, `Smem`, `Sync`, `TreeHalve`, `Cond`, atomic `Write` |
| matmul | plus `Loop`, `Init`, `Smem`, `CpAsyncCopy`/`Commit`/`Wait` |
| fused, attention | plus `Let`, `Accum` with `base` (online softmax), and in the f16 variants the tensor-core statements |

## Adding support for a missing op

The statement dispatcher is `emit_stmts` in `ptxgen/control.py`; anything it does not know raises
`NotImplementedError("statement <Name>")`. Index and predicate expressions are lowered in `ptxgen/expr.py`. To add an
op: write its `emit_*` function in the module that owns the concern, register it in the dispatcher, add a unit test,
and update the "What it supports" table in `COMPILER.md`.
