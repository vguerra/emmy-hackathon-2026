"""Tensor cores: mma.sync fragments, ldmatrix operand loads and the accumulator store.

Fragment layouts for mma.sync.aligned.m16n8k16 (f16 inputs, f32 accumulate), with
g = lane / 4 (group) and t = lane % 4 (thread in group):

    A (16x16, row)   4 regs, each two f16:  a0 (g,   2t..2t+1)   a1 (g+8, 2t..2t+1)
                                            a2 (g,   2t+8..+9)   a3 (g+8, 2t+8..+9)
    B (16x8,  col)   2 regs:                b0 (k 2t..2t+1, n g)  b1 (k 2t+8..+9, n g)
    C (16x8,  f32)   4 regs:                c0 (g, 2t)  c1 (g, 2t+1)  c2 (g+8, 2t)  c3 (g+8, 2t+1)

ldmatrix loads 8x8 b16 matrices; thread T of the warp receives row T/4, elements 2*(T%4)
and +1 of each (or the transpose with .trans). Lanes 8i..8i+7 supply the row addresses of
matrix i.
"""

import re

from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import (FragmentApply, FragmentRepack, FragmentRowReduce, LdmatrixLoad,
                                        MmaSyncPtx, RegFragment, RegStore)

from .dtypes import BOOL, F16, F32, emit_convert, emit_reg, emit_shfl_xor, f32_imm
from .elementwise import BINOPS, apply_op
from .emitter import PtxEmitter
from .expr import emit_flat_offset, emit_int_expr, emit_pred, int_op
from .memory import (buffer_shape, buffer_type, emit_address, emit_ld, emit_st, emit_swizzle, mem_space)

FRAGMENT_REF = re.compile(r"^(\w+)(?:\[(\w+)\])?$")
MMA_TYPES = {"f16": "f16"}      # bf16 would be one more entry, but there is no bf16 buffer to check it against


def fragment_regs(em: PtxEmitter, ref: str) -> list[str]:
    """Registers behind a fragment reference: `_a0`, or one member of a family, `_b[1]` / `_b[_r0]`."""
    match = FRAGMENT_REF.match(ref)
    if match is None:
        raise NotImplementedError(f"fragment reference {ref!r}")
    name, index = match.groups()
    regs = em.frags[name]
    if index is None:
        return regs
    member = int(index) if index.lstrip("-").isdigit() else int(em.vars[index])   # unrolled loop vars are immediates
    return regs[member]


def mma_nregs(role: str, shape: tuple, nbytes: int) -> int:
    """Registers one thread holds of an operand: elements per warp / 32 lanes, packed to 32 bits."""
    m, n, k = shape
    elems = {"a": m * k, "b": k * n, "c": m * n}[role]
    return elems // 32 if role == "c" else elems * nbytes // 4 // 32


def emit_reg_fragment(em: PtxEmitter, stmt: RegFragment) -> None:
    """Declare an operand array. C accumulators start at zero; A and B hold packed halves."""
    if stmt.role == "c" and stmt.dtype.nbytes != 4:
        raise NotImplementedError("f16-accumulate mma (packed C fragment)")
    if stmt.role != "c" and stmt.dtype.nbytes != 2:
        raise NotImplementedError(f"mma operand of {stmt.dtype.name}")
    count = stmt.nregs if stmt.nregs is not None else mma_nregs(stmt.role, stmt.shape, stmt.dtype.nbytes)

    def declare() -> list[str]:
        if stmt.role == "c":
            regs = [em.reg("f") for _ in range(count)]
            for reg in regs:
                em.emit(f"mov.f32 {reg}, 0f00000000;")
            return regs
        return [em.reg("r") for _ in range(count)]

    em.frags[stmt.name] = [declare() for _ in range(stmt.count)] if stmt.count > 1 else declare()


def lane_offset(em: PtxEmitter, stmt: LdmatrixLoad, ldm: int) -> str:
    """Element offset (row * ldm + col) of the smem row this lane addresses for the ldmatrix."""
    lane = em.vars["lane"]
    paired = stmt.role == "a" or stmt.pair_frag is not None       # x4: lanes cover 16 rows, two column blocks
    if paired:
        row = emit_reg(em, "r", f"and.b32 {lane}, 15")
        block = emit_reg(em, "r", f"shr.u32 {lane}, 4")
        col = emit_reg(em, "r", f"shl.b32 {block}, 3")
    elif stmt.b_trans:                                         # B stored N x K: x2, rows are n
        row = emit_reg(em, "r", f"and.b32 {lane}, 7")
        half = emit_reg(em, "r", f"shr.u32 {lane}, 3")
        col = emit_reg(em, "r", f"shl.b32 {emit_reg(em, 'r', f'and.b32 {half}, 1')}, 3")
    else:                                                      # K x 8 slab: x2.trans, rows are k
        row, col = emit_reg(em, "r", f"and.b32 {lane}, 15"), "0"
    return emit_reg(em, "r", f"mad.lo.s32 {row}, {ldm}, {col}")


def emit_direct_fragment(em: PtxEmitter, stmt: LdmatrixLoad, arg_types: dict) -> None:
    """Gather a fragment straight from global memory, one predicated 16-bit load per element.

    Used when the operand was not staged through shared memory. `k_zero` = (base, bound) zeroes
    the elements whose K coordinate (base + k) is at or past the bound, without reading them;
    `gmem_guard` = (base, bound) clamps the row (A) or column (B) to the tile that exists.
    """
    shape = buffer_shape(em, arg_types, stmt.src_buffer)
    if buffer_type(em, arg_types, stmt.src_buffer) is not F16:
        raise NotImplementedError("direct fragment load of a non-f16 buffer")
    ldm = stmt.ldm or shape[-1]
    base = emit_flat_offset(em, stmt.src_index, shape)
    lane = em.vars["lane"]
    group = emit_reg(em, "r", f"shr.u32 {lane}, 2")
    two_t = emit_reg(em, "r", f"shl.b32 {emit_reg(em, 'r', f'and.b32 {lane}, 3')}, 1")
    kmask = [emit_int_expr(em, e) for e in stmt.k_zero] if stmt.k_zero else None
    if stmt.gmem_guard:
        gbase, gbound = (emit_int_expr(em, e) for e in stmt.gmem_guard)
        last = emit_reg(em, "r", f"sub.s32 {emit_reg(em, 'r', f'sub.s32 {gbound}, {gbase}')}, 1")

    def element(mn_step: int, k_step: int, e: int) -> str:
        """Half-precision value of fragment element (mn = g + mn_step, k = 2t + k_step + e)."""
        mn = emit_reg(em, "r", f"add.s32 {group}, {mn_step}") if mn_step else group
        if stmt.gmem_guard:
            mn = emit_reg(em, "r", f"min.s32 {mn}, {last}")
        k = emit_reg(em, "r", f"add.s32 {two_t}, {k_step + e}")
        along_mn_first = stmt.role == "a" or stmt.b_trans          # A rows, or B stored N x K
        row, col = (mn, k) if along_mn_first else (k, mn)
        flat = emit_reg(em, "r", f"mad.lo.s32 {row}, {ldm}, {col}")
        addr = emit_address(em, stmt.src_buffer, emit_reg(em, "r", f"add.s32 {base}, {flat}"), 2)
        half = em.reg("h")
        em.emit(f"mov.b16 {half}, 0;")
        if kmask is None:
            em.emit(f"ld.global.b16 {half}, [{addr}];")
        else:
            valid = em.reg("p")
            em.emit(f"setp.lt.s32 {valid}, {emit_reg(em, 'r', f'add.s32 {kmask[0]}, {k}')}, {kmask[1]};")
            em.emit(f"@{valid} ld.global.b16 {half}, [{addr}];")
        return half

    regs = fragment_regs(em, stmt.frag)
    for i, reg in enumerate(regs):
        mn_step, k_step = ((8 * (i % 2), 8 * (i // 2)) if stmt.role == "a" else (0, 8 * i))
        low, high = element(mn_step, k_step, 0), element(mn_step, k_step, 1)
        em.emit(f"mov.b32 {reg}, {{{low}, {high}}};")


def emit_ldmatrix(em: PtxEmitter, stmt: LdmatrixLoad, arg_types: dict) -> None:
    """Load an A or B fragment: from shared memory with ldmatrix, else gathered from global memory."""
    if stmt.byte_slab or stmt.scale_buffer or stmt.fragment_layout != "m16n8k16":
        raise NotImplementedError("fp8, block-scaled or non-m16n8k16 fragment loads")
    if not stmt.staged:
        return emit_direct_fragment(em, stmt, arg_types)
    if stmt.gmem_guard or stmt.k_zero:
        raise NotImplementedError("guarded ldmatrix from shared memory")
    shape = buffer_shape(em, arg_types, stmt.src_buffer)
    ldm = stmt.ldm or shape[-1]
    base = emit_flat_offset(em, stmt.src_index, shape)
    element = emit_reg(em, "r", f"add.s32 {base}, {lane_offset(em, stmt, ldm)}")
    element = emit_swizzle(em, element, stmt.swizzle)
    addr = emit_address(em, stmt.src_buffer, element, 2)

    if stmt.role == "a":
        dest, shape_suffix, trans = fragment_regs(em, stmt.frag), "x4", ""
    elif stmt.pair_frag is not None:                           # two B fragments from one x4.trans
        dest = fragment_regs(em, stmt.frag) + fragment_regs(em, stmt.pair_frag)
        shape_suffix, trans = "x4", ".trans"
    else:
        dest, shape_suffix, trans = fragment_regs(em, stmt.frag), "x2", "" if stmt.b_trans else ".trans"
    em.emit(f"ldmatrix.sync.aligned.m8n8.{shape_suffix}{trans}.shared.b16 {{{', '.join(dest)}}}, [{addr}];")


def emit_mma(em: PtxEmitter, stmt: MmaSyncPtx) -> None:
    """c = a * b + c on the tensor cores, accumulating in place."""
    if stmt.c_dtype != "f32" or stmt.ab_dtype not in MMA_TYPES or stmt.block_scaled or stmt.b_row_major:
        raise NotImplementedError(f"mma {stmt.ab_dtype}->{stmt.c_dtype}")
    m, n, k = stmt.shape
    c, a, b = (fragment_regs(em, f) for f in (stmt.c_frag, stmt.a_frag, stmt.b_frag))
    ab = MMA_TYPES[stmt.ab_dtype]
    em.emit(f"mma.sync.aligned.m{m}n{n}k{k}.row.col.f32.{ab}.{ab}.f32 "
            f"{{{', '.join(c)}}}, {{{', '.join(a)}}}, {{{', '.join(b)}}}, {{{', '.join(c)}}};")


def emit_reg_store(em: PtxEmitter, stmt: RegStore, arg_types: dict) -> None:
    """Write a C fragment: each lane stores its two row pairs (g and g+8) of two columns."""
    if stmt.fragment_layout != "m16n8k16":
        raise NotImplementedError(f"RegStore of fragment layout {stmt.fragment_layout}")
    if stmt.epilogue is not None:
        passed = stmt.epilogue.params[1 + len(stmt.extra_frags):]       # after the accumulators: coordinates, axes
        unknown = [name for name in passed if name not in ("__M__", "__N__") and name not in em.vars]
        if len(stmt.epilogue.params) < 1 + len(stmt.extra_frags) or unknown:
            raise NotImplementedError(f"epilogue parameters {unknown or list(stmt.epilogue.params)}")
    t = buffer_type(em, arg_types, stmt.dst_buffer)
    shape = buffer_shape(em, arg_types, stmt.dst_buffer)
    row_dim = stmt.row_dim if stmt.row_dim is not None else len(shape) - 2
    ldm = stmt.ldm or _prod(shape[row_dim + 1:])
    lane = em.vars["lane"]
    group = emit_reg(em, "r", f"shr.u32 {lane}, 2")
    pair = emit_reg(em, "r", f"shl.b32 {emit_reg(em, 'r', f'and.b32 {lane}, 3')}, 1")
    base = emit_flat_offset(em, stmt.dst_index, shape)
    paired = pair_store_ok(stmt, ldm, t)
    for half in (0, 1):                                        # rows g and g + 8
        row = emit_reg(em, "r", f"mad.lo.s32 {group}, {ldm}, {pair}")
        element = emit_reg(em, "r", f"add.s32 {row}, {base}")
        if half:
            element = emit_reg(em, "r", f"add.s32 {element}, {8 * ldm}")
        if paired:
            _store_columns(em, stmt, arg_types, t, element, (0, 1), group, pair, half)
        else:                                                  # each column on its own: own bounds check and address
            for e in (0, 1):
                _store_columns(em, stmt, arg_types, t, element, (e,), group, pair, half)


def pair_store_ok(stmt: RegStore, ldm: int, t) -> bool:
    """True when a lane's two adjacent columns may be written as one 2-element access.

    A 2-wide access needs a 2-element-aligned address, so the row stride must be even, and a masked tile
    must never end between the pair's columns, so the column bound must be a known even number.
    """
    if ldm % 2:
        return False
    if stmt.n_guard is None:
        return True
    bound = stmt.n_guard[1]
    return isinstance(bound, Literal) and int(bound.value) % 2 == 0


def _store_columns(em: PtxEmitter, stmt: RegStore, arg_types: dict, t, element: str, columns: tuple,
                   group: str, pair: str, half: int) -> None:
    """Store the fragment elements of `columns` (0, 1 or both) of one row, as one access."""
    skip = _guard(em, stmt, group, pair, half, columns[-1])
    first = columns[0]
    where = emit_swizzle(em, int_op(em, "+", element, str(first)), stmt.swizzle)
    addr = emit_address(em, stmt.dst_buffer, where, t.nbytes)
    values = [_stored_value(em, stmt, arg_types, t, 2 * half + e, group, pair, half, e) for e in columns]
    if stmt.atomic:
        _emit_atomic(em, addr, t, values)
    else:
        emit_st(em, mem_space(em, stmt.dst_buffer), addr, t, values)
    if skip:
        em.label(skip)


def _stored_value(em: PtxEmitter, stmt: RegStore, arg_types: dict, t, index: int, group: str,
                  pair: str, half: int, e: int) -> str:
    """Register to store for fragment element `index` (c0..c3), after the fused epilogue if any."""
    target = F32 if t is F16 else t                            # f16 is narrowed inside the store
    reg = fragment_regs(em, stmt.frag)[index]
    if stmt.epilogue is None:
        return emit_convert(em, reg, F32, target)
    from .control import emit_stmts                            # the epilogue body is ordinary statements

    em.bind(stmt.epilogue.params[0], reg, F32)
    for param, frag in zip(stmt.epilogue.params[1:], stmt.extra_frags):
        em.bind(param, fragment_regs(em, frag)[index], F32)
    # ELEM_ROW / ELEM_COL: this element's offsets inside the 16x8 tile
    em.vars["__M__"] = emit_reg(em, "r", f"add.s32 {group}, {8 * half}") if half else group
    em.vars["__N__"] = emit_reg(em, "r", f"add.s32 {pair}, {e}") if e else pair
    emit_stmts(em, stmt.epilogue.body, arg_types)
    result = stmt.epilogue.results[0]
    return emit_convert(em, em.vars[result], em.type_of(result), target)


def _prod(values) -> int:
    result = 1
    for v in values:
        result *= v
    return result


def _guard(em: PtxEmitter, stmt: RegStore, group: str, pair: str, half: int, last_column: int) -> str | None:
    """Branch past a store when its row or its last column lies outside a masked tile; returns the label."""
    if stmt.m_guard is None and stmt.n_guard is None:
        return None
    skip = em.new_label("STORE_SKIP")
    if stmt.m_guard is not None:
        base, bound = (emit_int_expr(em, e) for e in stmt.m_guard)
        row = emit_reg(em, "r", f"add.s32 {base}, {group}")
        if half:
            row = emit_reg(em, "r", f"add.s32 {row}, 8")
        out = em.reg("p")
        em.emit(f"setp.ge.s32 {out}, {row}, {bound};")
        em.emit(f"@{out} bra {skip};")
    if stmt.n_guard is not None:                               # every column the access covers must fit
        base, bound = (emit_int_expr(em, e) for e in stmt.n_guard)
        col = emit_reg(em, "r", f"add.s32 {base}, {pair}")
        last = int_op(em, "+", col, str(last_column))
        out = em.reg("p")
        em.emit(f"setp.ge.s32 {out}, {last}, {bound};")
        em.emit(f"@{out} bra {skip};")
    return skip


def _emit_atomic(em: PtxEmitter, addr: str, t, values: list[str]) -> None:
    """Atomic add of one or two adjacent elements: f16 pairs are packed, anything else is one instruction each."""
    if t is F16 and len(values) == 1:
        half = emit_reg(em, "h", f"cvt.rn.f16.f32 {values[0]}")
        em.emit(f"red.global.add.noftz.f16 [{addr}], {half};")
        return
    if t is F16:
        halves = [emit_reg(em, "h", f"cvt.rn.f16.f32 {v}") for v in values]
        word = em.reg("r")
        em.emit(f"mov.b32 {word}, {{{halves[0]}, {halves[1]}}};")
        em.emit(f"red.global.add.noftz.f16x2 [{addr}], {word};")
        return
    for i, value in enumerate(values):
        em.emit(f"red.global.add.{t.mem} [{addr}+{t.nbytes * i}], {value};")


# ---- fragment-level pointwise, row reductions and the C -> A repack (fused attention) ----------

def element_coords(em: PtxEmitter, stmt: FragmentApply, i: int, group: str, quad: str) -> None:
    """Bind __frow / __fcol to the absolute coordinates of fragment element i (tile origin + layout offset)."""
    em.vars["_g"], em.vars["_t"] = group, quad
    row = emit_int_expr(em, stmt.layout.row_off[stmt.layout.elem_row[i]])
    col = emit_int_expr(em, stmt.layout.col_off[i])
    if stmt.row_base is not None:
        row = emit_reg(em, "r", f"add.s32 {emit_int_expr(em, stmt.row_base)}, {row}")
    if stmt.col_base is not None:
        col = emit_reg(em, "r", f"add.s32 {emit_int_expr(em, stmt.col_base)}, {col}")
    em.vars["__frow"], em.vars["__fcol"] = row, col


def fragment_operand(em: PtxEmitter, stmt: FragmentApply, arg_types: dict, arg, kind: str, i: int) -> tuple:
    """(register, Ty) of one FragmentApply argument as seen by fragment element i."""
    if kind == "frag":
        return fragment_regs(em, arg)[i], F32
    if kind == "row":                                          # one scalar per row, shared by the row's elements
        name = arg[stmt.layout.elem_row[i]]
        return em.vars[name], em.type_of(name)
    if kind == "uniform":
        if isinstance(arg, str):
            return em.vars[arg], em.type_of(arg)
        return emit_reg(em, "f", f"mov.f32 {f32_imm(float(arg.value))}"), F32
    if kind == "coord":                                        # a predicate over absolute coordinates
        return emit_pred(em, arg), BOOL
    if kind == "gmem":                                         # a load at the element's coordinates
        buffer, index = arg
        t = buffer_type(em, arg_types, buffer)
        offset = emit_flat_offset(em, index, buffer_shape(em, arg_types, buffer))
        addr = emit_address(em, buffer, offset, t.nbytes)
        return emit_ld(em, mem_space(em, buffer), addr, t, 1)[0], F32
    raise NotImplementedError(f"FragmentApply operand kind {kind!r}")


def emit_fragment_apply(em: PtxEmitter, stmt: FragmentApply, arg_types: dict) -> None:
    """out[i] = op(args...) for every element i of a C fragment, then any `post` unary ops."""
    layout = stmt.layout
    if layout.n_elems != 4:
        raise NotImplementedError("FragmentApply on a fragment layout other than m16n8")
    lane = em.vars["lane"]
    group = emit_reg(em, "r", f"shr.u32 {lane}, 2")
    quad = emit_reg(em, "r", f"and.b32 {lane}, 3")
    results = []
    for i in range(layout.n_elems):
        if any(k in ("coord", "gmem") for k in stmt.kinds):
            element_coords(em, stmt, i, group, quad)
        args = [fragment_operand(em, stmt, arg_types, a, k, i) for a, k in zip(stmt.args, stmt.kinds)]
        value, t = apply_op(em, stmt.op.name, args, F32)
        for post in stmt.post:
            value, t = apply_op(em, post.name, [(value, t)], F32)
        results.append(value)
    if stmt.in_place:                                          # write back into the fragment's own registers
        for reg, value in zip(fragment_regs(em, stmt.out), results):
            em.emit(f"mov.f32 {reg}, {value};")
    else:
        em.frags[stmt.out] = results


def emit_fragment_row_reduce(em: PtxEmitter, stmt: FragmentRowReduce) -> None:
    """Fold each row over the chunk's fragments, then across the lanes that hold the row's columns."""
    layout = stmt.layout
    if layout.n_elems != 4:
        raise NotImplementedError("FragmentRowReduce on a fragment layout other than m16n8")
    combine = BINOPS["f"][stmt.op.name]
    for name, row in ((stmt.top, 0), (stmt.bot, 1)):
        parts = [fragment_regs(em, f)[i] for f in stmt.frags for i in range(layout.n_elems)
                 if layout.elem_row[i] == row]
        total = parts[0]
        for part in parts[1:]:
            total = emit_reg(em, "f", f"{combine} {total}, {part}")
        for lanes in layout.reduce_xors:
            other = emit_reg(em, "f", f"mov.b32 {emit_shfl_xor(em, total, lanes)}")
            total = emit_reg(em, "f", f"{combine} {total}, {other}")
        em.bind(name, total, F32)


def emit_fragment_repack(em: PtxEmitter, stmt: FragmentRepack) -> None:
    """Two k-adjacent C fragments become one A fragment: each pair of f32 packs into one b32 of two halves.

    The lanes of an m16n8 C fragment already hold what an m16n8k16 A fragment needs, so no data
    moves between lanes: a0, a1 come from the first source (rows g, g+8) and a2, a3 from the second.
    """
    if stmt.fragment_layout != "m16n8k16" or stmt.ab_dtype not in MMA_TYPES:
        raise NotImplementedError(f"FragmentRepack {stmt.ab_dtype} / {stmt.fragment_layout}")
    srcs = stmt.srcs[2 * stmt.part: 2 * stmt.part + 2] if len(stmt.srcs) > 2 else stmt.srcs
    if len(srcs) != 2:
        raise NotImplementedError(f"FragmentRepack of {len(stmt.srcs)} sources")
    packed = MMA_TYPES[stmt.ab_dtype]
    regs = fragment_regs(em, stmt.frag)
    for index, (src, first) in enumerate(((srcs[0], 0), (srcs[0], 2), (srcs[1], 0), (srcs[1], 2))):
        c = fragment_regs(em, src)
        em.emit(f"cvt.rn.{packed}x2.f32 {regs[index]}, {c[first + 1]}, {c[first]};")   # first operand is the high half
