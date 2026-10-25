"""Element types and the conversions between them.

Every SSA value lives in a register of one *class*:

    f   .f32   holds f32, and also f16 values (computed in f32, rounded back after each op)
    d   .f64   f64
    r   .b32   i32
    rd  .b64   i64
    p   .pred  booleans (comparison results)

A narrow type is therefore a storage format, not a register format: f16 is widened on load,
rounded on every Assign whose result type is f16, and narrowed on store. This is how CUDA
evaluates __half arithmetic, so it matches the reference kernels.
"""

import math
import struct
from dataclasses import dataclass

from .emitter import PtxEmitter


@dataclass(frozen=True)
class Ty:
    name: str       # IR / meta.json name
    nbytes: int     # size of one element in memory
    cls: str        # register class that holds a value of this type
    ptx: str        # type suffix for arithmetic, comparisons and cvt
    mem: str        # type suffix for ld / st


F16_MAX = 65504.0

F32 = Ty("f32", 4, "f", "f32", "f32")
F16 = Ty("f16", 2, "f", "f32", "b16")
F64 = Ty("f64", 8, "d", "f64", "f64")
I32 = Ty("i32", 4, "r", "s32", "s32")
I64 = Ty("i64", 8, "rd", "s64", "s64")
BOOL = Ty("bool", 1, "p", "pred", "pred")

TYPES = {t.name: t for t in (F32, F16, F64, I32, I64, BOOL)}

# Smem.dtype is spelled the CUDA way
SMEM_NAMES = {"float": "f32", "half": "f16", "__half": "f16", "double": "f64", "int": "i32", "long long": "i64",
               "unsigned long long": "i64"}   # an mbarrier is one 64-bit word


def ty(name) -> Ty:
    """The Ty for an IR dtype: a name string, or an emmy DataType (which has .name)."""
    name = getattr(name, "name", name)
    name = SMEM_NAMES.get(name, name)
    if name not in TYPES:
        raise NotImplementedError(f"data type {name!r}")
    return TYPES[name]


def f32_imm(value: float) -> str:
    """PTX hex spelling of an f32 literal."""
    return "0f%08X" % struct.unpack(">I", struct.pack(">f", value))[0]


def f64_imm(value: float) -> str:
    """PTX hex spelling of an f64 literal."""
    return "0d%016X" % struct.unpack(">Q", struct.pack(">d", value))[0]


def imm(t: Ty, value: float) -> str:
    """An immediate operand of type `t`."""
    if t is F16:                                        # the value an f16 would actually hold
        value = math.copysign(math.inf, value) if abs(value) > F16_MAX else value
        value = struct.unpack("<e", struct.pack("<e", float(value)))[0]
    if t.cls == "f":
        return f32_imm(float(value))
    if t.cls == "d":
        return f64_imm(float(value))
    return str(int(value))


def identity_imm(t: Ty, op: str, identity: float) -> str:
    """Neutral element of a fold. Emmy states max/min identities as +-1e30, which is wrong for ints."""
    if t.cls in ("r", "rd") and abs(identity) >= 1e30:
        bits = 8 * t.nbytes - 1
        return str(-(2 ** bits) if identity < 0 else 2 ** bits - 1)
    return imm(t, identity)


def emit_reg(em: PtxEmitter, cls: str, line: str) -> str:
    """Emit `<opcode> dst, operands;` into a fresh register of class `cls` and return it."""
    dst = em.reg(cls)
    opcode, operands = line.split(" ", 1)
    em.emit(f"{opcode} {dst}, {operands};")
    return dst


def round_f16(em: PtxEmitter, x: str) -> str:
    """x rounded to the nearest f16, still held in an f32 register."""
    half = emit_reg(em, "h", f"cvt.rn.f16.f32 {x}")
    return emit_reg(em, "f", f"cvt.f32.f16 {half}")


def emit_convert(em: PtxEmitter, x: str, src: Ty, dst: Ty) -> str:
    """Register holding `x` (of type src) as type dst; `x` itself when no work is needed."""
    if src.cls == dst.cls:
        return round_f16(em, x) if dst is F16 and src is not F16 and src.cls == "f" else x
    src_float, dst_float = src.cls in ("f", "d"), dst.cls in ("f", "d")
    if dst_float:
        rounding = ".rn" if (src_float and dst.cls == "f") or not src_float else ""
        converted = emit_reg(em, dst.cls, f"cvt{rounding}.{dst.ptx}.{src.ptx} {x}")
        return round_f16(em, converted) if dst is F16 else converted
    rounding = ".rzi" if src_float else ""
    return emit_reg(em, dst.cls, f"cvt{rounding}.{dst.ptx}.{src.ptx} {x}")


def emit_shfl_xor(em: PtxEmitter, x: str, lanes: int) -> str:
    """The value of `x` held by lane (self ^ lanes), a warp-wide butterfly step on a 32-bit register."""
    bits, shuffled = em.reg("r"), em.reg("r")
    em.emit(f"mov.b32 {bits}, {x};")
    em.emit(f"shfl.sync.bfly.b32 {shuffled}, {bits}, {lanes}, 0x1f, 0xffffffff;")
    return shuffled
