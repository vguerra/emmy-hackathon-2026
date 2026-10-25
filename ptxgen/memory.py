"""Memory statements: global and shared addressing, Load, Write, cp.async and Smem."""

import math

from emmy.compiler.ir.kernel.ir import CpAsyncCopy, Smem
from emmy.compiler.ir.stmt.leaves import Load, Write

from .dtypes import F16, Ty, emit_convert, emit_reg, F32, ty
from .emitter import PtxEmitter, is_imm
from .expr import emit_flat_offset, emit_int_expr, int_op

# Vector widths PTX allows per element size. Eight f16 values are not a vector type: they are moved as
# four packed 32-bit words (see emit_ld / emit_st).
VECTOR_WIDTHS = {2: (1, 2, 4), 4: (1, 2, 4), 8: (1, 2)}
F16_WORD_PACKED_WIDTH = 8
# Atomic add spellings; integers use the unsigned form, which wraps identically.
RED_TYPES = {"f32": "f32", "f64": "f64", "f16": "noftz.f16", "i32": "u32", "i64": "u64"}


# Software smem swizzle: element index e -> e ^ (((e >> shift) & mask) << 3), i.e. the 16-byte
# chunks within each group of rows are permuted. Fill (cp.async) and drain (ldmatrix) apply the
# same address-based XOR, so they agree by construction.
SWIZZLES = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}


def swizzle_params(mode: str) -> tuple[int, int] | None:
    """(element shift, row mask) of a swizzle mode, 'B64@5' overriding the shift; None for 'NONE'."""
    base, _, shift = mode.partition("@")
    if base == "NONE":
        return None
    if base not in SWIZZLES:
        raise NotImplementedError(f"smem swizzle {mode!r}")
    default_shift, mask = SWIZZLES[base]
    return (int(shift) if shift else default_shift), mask


def emit_swizzle(em: PtxEmitter, element: str, mode: str) -> str:
    """The swizzled element index of `element` (unchanged for mode NONE)."""
    params = swizzle_params(mode)
    if params is None:
        return element
    shift, mask = params
    field = emit_reg(em, "r", f"and.b32 {emit_reg(em, 'r', f'shr.u32 {element}, {shift}')}, {mask}")
    return emit_reg(em, "r", f"xor.b32 {element}, {emit_reg(em, 'r', f'shl.b32 {field}, 3')}")


def buffer_shape(em: PtxEmitter, arg_types: dict, buffer: str) -> list[int]:
    """Extents of a kernel argument or of a shared array."""
    return list(em.smem_shapes[buffer]) if buffer in em.smem_base else arg_types[buffer][1]


def buffer_type(em: PtxEmitter, arg_types: dict, buffer: str) -> Ty:
    """Element type of a kernel argument (from meta.json) or of a shared array."""
    return em.smem_types[buffer] if buffer in em.smem_base else ty(arg_types[buffer][0])


def mem_space(em: PtxEmitter, buffer: str) -> str:
    return ".shared" if buffer in em.smem_base else ".global"


def emit_address(em: PtxEmitter, buffer: str, offset: str, elem_bytes: int) -> str:
    """Byte address of element `offset` of `buffer`: 32-bit for shared, 64-bit for global."""
    if buffer in em.smem_base:
        return int_op(em, "+", em.smem_base[buffer], int_op(em, "*", offset, str(elem_bytes)))
    if is_imm(offset):
        return emit_reg(em, "rd", f"add.s64 {em.buffers[buffer]}, {int(offset) * elem_bytes}")
    wide = emit_reg(em, "rd", f"mul.wide.s32 {offset}, {elem_bytes}")
    return emit_reg(em, "rd", f"add.s64 {em.buffers[buffer]}, {wide}")


def vector_suffix(n: int, nbytes: int) -> str:
    """PTX vector qualifier for an n-wide access to elements of `nbytes`: '', '.v2' or '.v4'."""
    if n not in VECTOR_WIDTHS[nbytes]:
        raise NotImplementedError(f"{n}-wide access to {nbytes}-byte elements")
    return "" if n == 1 else f".v{n}"


def emit_ld(em: PtxEmitter, space: str, addr: str, t: Ty, n: int) -> list[str]:
    """Load n consecutive elements as one instruction; returns their registers (f16 widened to f32)."""
    if t is not F16:
        regs = [em.reg(t.cls) for _ in range(n)]
        operands = f"{{{', '.join(regs)}}}" if n > 1 else regs[0]
        em.emit(f"ld{space}{vector_suffix(n, t.nbytes)}.{t.mem} {operands}, [{addr}];")
        return regs
    if n == F16_WORD_PACKED_WIDTH:                      # 8 halves = 4 packed 32-bit words
        words = [em.reg("r") for _ in range(4)]
        em.emit(f"ld{space}.v4.b32 {{{', '.join(words)}}}, [{addr}];")
        halves = []
        for word in words:
            pair = [em.reg("h"), em.reg("h")]
            em.emit(f"mov.b32 {{{pair[0]}, {pair[1]}}}, {word};")
            halves += pair
    else:
        halves = [em.reg("h") for _ in range(n)]
        operands = f"{{{', '.join(halves)}}}" if n > 1 else halves[0]
        em.emit(f"ld{space}{vector_suffix(n, 2)}.b16 {operands}, [{addr}];")
    return [emit_reg(em, "f", f"cvt.f32.f16 {h}") for h in halves]


def emit_st(em: PtxEmitter, space: str, addr: str, t: Ty, values: list[str]) -> None:
    """Store `values` (registers of class t.cls) as one instruction; f16 is narrowed first."""
    n = len(values)
    if t is not F16:
        operands = f"{{{', '.join(values)}}}" if n > 1 else values[0]
        em.emit(f"st{space}{vector_suffix(n, t.nbytes)}.{t.mem} [{addr}], {operands};")
        return
    halves = [emit_reg(em, "h", f"cvt.rn.f16.f32 {v}") for v in values]
    if n == F16_WORD_PACKED_WIDTH:
        words = []
        for i in range(0, 8, 2):
            word = em.reg("r")
            em.emit(f"mov.b32 {word}, {{{halves[i]}, {halves[i + 1]}}};")
            words.append(word)
        em.emit(f"st{space}.v4.b32 [{addr}], {{{', '.join(words)}}};")
    else:
        operands = f"{{{', '.join(halves)}}}" if n > 1 else halves[0]
        em.emit(f"st{space}{vector_suffix(n, 2)}.b16 [{addr}], {operands};")


def emit_load(em: PtxEmitter, stmt: Load, arg_types: dict) -> None:
    """One value, or several consecutive values as a vector load."""
    t = buffer_type(em, arg_types, stmt.input)
    shape = buffer_shape(em, arg_types, stmt.input)
    addr = emit_address(em, stmt.input, emit_flat_offset(em, stmt.index, shape), t.nbytes)
    regs = emit_ld(em, mem_space(em, stmt.input), addr, t, len(stmt.names))
    for name, reg in zip(stmt.names, regs):
        em.bind(name, reg, t)


def emit_write(em: PtxEmitter, stmt: Write, arg_types: dict) -> None:
    """Store one value or a vector; an atomic write accumulates with red.global.add."""
    t = buffer_type(em, arg_types, stmt.output)
    shape = buffer_shape(em, arg_types, stmt.output)
    addr = emit_address(em, stmt.output, emit_flat_offset(em, stmt.index, shape), t.nbytes)
    space = mem_space(em, stmt.output)
    values = []
    for name in stmt.values:                            # convert each value to the buffer's type
        source = em.type_of(name)
        target = F32 if t is F16 else t                 # f16 values are narrowed inside emit_st
        values.append(emit_convert(em, em.vars[name], source, target))
    if stmt.atomic:                                     # no vector red on sm_89: one per element
        for i, value in enumerate(values):
            if t is F16:
                value = emit_reg(em, "h", f"cvt.rn.f16.f32 {value}")
            em.emit(f"red.global.add.{RED_TYPES[t.name]} [{addr}+{t.nbytes * i}], {value};")
    else:
        emit_st(em, space, addr, t, values)


def emit_cp_async(em: PtxEmitter, stmt: CpAsyncCopy, arg_types: dict) -> None:
    """Global -> shared copy of 4/8/16 bytes; 16-byte copies must use .cg."""
    if stmt.nbytes not in (4, 8, 16):
        raise NotImplementedError(f"cp.async of {stmt.nbytes} bytes")
    smem_shape = buffer_shape(em, arg_types, stmt.smem)
    smem_type = buffer_type(em, arg_types, stmt.smem)
    src_type = buffer_type(em, arg_types, stmt.src)
    offset = emit_flat_offset(em, stmt.smem_index, smem_shape)
    if stmt.lane_index is not None:                     # this thread's own (row, col) within the stripe
        row, col = (emit_int_expr(em, e) for e in stmt.lane_index)
        lane = em.reg("r")
        em.emit(f"mad.lo.s32 {lane}, {row}, {smem_shape[-1]}, {col};")
        total = em.reg("r")
        em.emit(f"add.s32 {total}, {offset}, {lane};")
        offset = total
    dst = emit_address(em, stmt.smem, emit_swizzle(em, offset, stmt.swizzle), smem_type.nbytes)
    src_offset = emit_flat_offset(em, stmt.src_index, buffer_shape(em, arg_types, stmt.src))
    src = emit_address(em, stmt.src, src_offset, src_type.nbytes)
    cache = "cg" if stmt.nbytes == 16 else "ca"
    em.emit(f"cp.async.{cache}.shared.global [{dst}], [{src}], {stmt.nbytes};")


def emit_smem(em: PtxEmitter, stmt: Smem) -> None:
    """Declare a shared array: a static .shared, or a slice of the one extern array."""
    t = ty(stmt.dtype)
    align = max(stmt.align, t.nbytes)
    nbytes = t.nbytes * math.prod(stmt.extents)
    em.smem_base[stmt.name] = em.reg("r")
    em.smem_shapes[stmt.name] = stmt.extents
    em.smem_types[stmt.name] = t
    if em.dynamic_smem:                                 # carve the array out of the extern one
        em.smem_used = -(-em.smem_used // align) * align
        em.emit(f"mov.u32 {em.smem_base[stmt.name]}, smem_dyn;")
        if em.smem_used:
            em.emit(f"add.s32 {em.smem_base[stmt.name]}, {em.smem_base[stmt.name]}, {em.smem_used};")
        em.smem_used += nbytes
    else:
        em.shared.append(f".shared .align {align} .b8 {stmt.name}[{nbytes}];")
        em.emit(f"mov.u32 {em.smem_base[stmt.name]}, {stmt.name};")
