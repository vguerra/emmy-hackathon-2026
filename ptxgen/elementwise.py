"""Scalar arithmetic: Assign (by op name and operand type), Let and Select."""

import math

from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.leaves import Assign, Let, Select

from .dtypes import (BOOL, F16, F32, I32, Ty, emit_convert, emit_reg, f32_imm, imm, round_f16, ty)
from .emitter import PtxEmitter
from .expr import emit_int_expr, emit_pred

# Arithmetic by register class (see dtypes.py). .rn pins the rounding mode, which also stops
# ptxas from fusing mul+add into fma (the expected outputs come from numpy, which rounds
# every operation).
BINOPS = {
    "f": {"add": "add.rn.f32", "subtract": "sub.rn.f32", "multiply": "mul.rn.f32",
          "divide": "div.rn.f32", "maximum": "max.f32", "minimum": "min.f32"},
    "d": {"add": "add.rn.f64", "subtract": "sub.rn.f64", "multiply": "mul.rn.f64",
          "divide": "div.rn.f64", "maximum": "max.f64", "minimum": "min.f64"},
    "r": {"add": "add.s32", "subtract": "sub.s32", "multiply": "mul.lo.s32",
          "maximum": "max.s32", "minimum": "min.s32"},
    "rd": {"add": "add.s64", "subtract": "sub.s64", "multiply": "mul.lo.s64",
           "maximum": "max.s64", "minimum": "min.s64"},
}
UNOPS = {
    "f": {"sqrt": "sqrt.rn.f32", "reciprocal": "rcp.rn.f32", "rsqrt": "rsqrt.approx.f32",
          "abs": "abs.f32", "negative": "neg.f32", "exp2": "ex2.approx.f32",
          "log2": "lg2.approx.f32"},
    "d": {"sqrt": "sqrt.rn.f64", "reciprocal": "rcp.rn.f64", "abs": "abs.f64",
          "negative": "neg.f64"},
    "r": {"abs": "abs.s32", "negative": "neg.s32"},
    "rd": {"abs": "abs.s64", "negative": "neg.s64"},
}
# Without .rn ptxas may contract a multiply feeding an add into one fma, which rounds once instead
# of twice. Allowed only when the problem's tolerance is non-zero (see compile_ir).
CONTRACTABLE = {"add": "add.f32", "subtract": "sub.f32", "multiply": "mul.f32"}
MOVES = {"f": "mov.f32", "d": "mov.f64", "r": "mov.s32", "rd": "mov.s64"}
SELECTS = {"f": "selp.f32", "d": "selp.f64", "r": "selp.s32", "rd": "selp.s64"}
COMPARES = {"greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le",
            "equal": "eq", "not_equal": "ne"}
CMP_TYPES = {"f": "f32", "d": "f64", "r": "s32", "rd": "s64"}


def emit_f32(em: PtxEmitter, line: str) -> str:
    """Emit `<opcode> dst, operands;` into a fresh f32 register and return it."""
    return emit_reg(em, "f", line)


def emit_exp(em: PtxEmitter, x: str, scale: float = 1.0) -> str:
    """e^(scale * x) = 2^(scale * x * log2 e)."""
    scaled = emit_f32(em, f"mul.f32 {x}, {f32_imm(scale * math.log2(math.e))}")
    return emit_f32(em, f"ex2.approx.f32 {scaled}")


def emit_log(em: PtxEmitter, x: str) -> str:
    """ln x = log2 x * ln 2."""
    base2 = emit_f32(em, f"lg2.approx.f32 {x}")
    return emit_f32(em, f"mul.f32 {base2}, {f32_imm(math.log(2.0))}")


def emit_sigmoid(em: PtxEmitter, x: str) -> str:
    """1 / (1 + e^-x)."""
    denom = emit_f32(em, f"add.rn.f32 {emit_exp(em, x, -1.0)}, {f32_imm(1.0)}")
    return emit_f32(em, f"rcp.rn.f32 {denom}")


def emit_tanh(em: PtxEmitter, x: str) -> str:
    """tanh x = 1 - 2 / (e^(2x) + 1)."""
    denom = emit_f32(em, f"add.rn.f32 {emit_exp(em, x, 2.0)}, {f32_imm(1.0)}")
    half = emit_f32(em, f"rcp.rn.f32 {denom}")
    twice = emit_f32(em, f"mul.rn.f32 {half}, {f32_imm(2.0)}")
    return emit_f32(em, f"sub.rn.f32 {f32_imm(1.0)}, {twice}")


def emit_erf(em: PtxEmitter, x: str) -> str:
    """Abramowitz-Stegun 7.1.26: erf |x| = 1 - poly(t) * e^(-x^2), t = 1 / (1 + p|x|); abs error 1.5e-7."""
    p, a1, a2, a3, a4, a5 = 0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    magnitude = emit_f32(em, f"abs.f32 {x}")
    t = emit_f32(em, f"rcp.rn.f32 {emit_f32(em, f'fma.rn.f32 {magnitude}, {f32_imm(p)}, {f32_imm(1.0)}')}")
    poly = emit_f32(em, f"fma.rn.f32 {f32_imm(a5)}, {t}, {f32_imm(a4)}")
    for coeff in (a3, a2, a1):
        poly = emit_f32(em, f"fma.rn.f32 {poly}, {t}, {f32_imm(coeff)}")
    poly = emit_f32(em, f"mul.rn.f32 {poly}, {t}")
    decay = emit_exp(em, emit_f32(em, f"mul.rn.f32 {x}, {x}"), -1.0)
    tail = emit_f32(em, f"mul.rn.f32 {poly}, {decay}")
    return emit_f32(em, f"copysign.f32 {x}, {emit_f32(em, f'sub.rn.f32 {f32_imm(1.0)}, {tail}')}")


def emit_gelu(em: PtxEmitter, x: str) -> str:
    """0.5 x (1 + erf(x / sqrt 2))."""
    scaled = emit_f32(em, f"mul.rn.f32 {x}, {f32_imm(math.sqrt(0.5))}")
    one_plus = emit_f32(em, f"add.rn.f32 {emit_erf(em, scaled)}, {f32_imm(1.0)}")
    return emit_f32(em, f"mul.rn.f32 {emit_f32(em, f'mul.rn.f32 {x}, {f32_imm(0.5)}')}, {one_plus}")


def emit_softplus(em: PtxEmitter, x: str) -> str:
    """ln(1 + e^x)."""
    return emit_log(em, emit_f32(em, f"add.rn.f32 {emit_exp(em, x)}, {f32_imm(1.0)}"))


# Unary f32 ops built from several instructions: name -> emitter returning the result register.
F32_COMPOSED = {
    "exp": emit_exp,
    "log": emit_log,
    "sigmoid": emit_sigmoid,
    "tanh": emit_tanh,
    "erf": emit_erf,
    "gelu": emit_gelu,
    "softplus": emit_softplus,
    "silu": lambda em, x: emit_f32(em, f"mul.rn.f32 {x}, {emit_sigmoid(em, x)}"),
}


def binop(em: PtxEmitter, cls: str, op: str) -> str:
    """Opcode for binary `op` in register class `cls`, contractable when the tolerance allows."""
    if cls == "f" and em.allow_fma and op in CONTRACTABLE:
        return CONTRACTABLE[op]
    return BINOPS[cls][op]


def operand(em: PtxEmitter, name: str, target: Ty) -> str:
    """Register of SSA value `name`, converted when its register class differs from `target`'s.
    A same-class value is used as is: an f32 operand of an f16 op is not rounded to f16."""
    return as_class(em, em.vars[name], em.type_of(name), target)


def as_class(em: PtxEmitter, reg: str, source: Ty, target: Ty) -> str:
    """`reg` unchanged when it already lives in `target`'s register class, else converted."""
    return reg if source.cls == target.cls else emit_convert(em, reg, source, target)


def apply_op(em: PtxEmitter, op: str, args: list, result: Ty = None) -> tuple[str, Ty]:
    """Apply elementwise op `op` to `args`, a list of (register, Ty); returns (register, Ty).

    The op runs in the register class of its first (value) operand, or of the second for
    `where`; a result type of f16 is rounded afterwards, and one of another class is converted.
    Comparisons yield a predicate register.
    """
    kind = args[1][1] if op == "where" else args[0][1]
    cls = kind.cls
    result = result or kind
    regs = [as_class(em, reg, t, kind) for reg, t in args] if op != "where" else None
    if op in COMPARES:
        pred = em.reg("p")
        em.emit(f"setp.{COMPARES[op]}.{CMP_TYPES[cls]} {pred}, {regs[0]}, {regs[1]};")
        return pred, BOOL
    if op == "where":
        (cond, _), (a, ta), (b, tb) = args
        dst = emit_reg(em, cls, f"{SELECTS[cls]} {as_class(em, a, ta, kind)}, {as_class(em, b, tb, kind)}, {cond}")
    elif op in BINOPS[cls]:
        dst = emit_reg(em, cls, f"{binop(em, cls, op)} {regs[0]}, {regs[1]}")
    elif op in UNOPS[cls]:
        dst = emit_reg(em, cls, f"{UNOPS[cls][op]} {regs[0]}")
    elif cls == "f" and op in F32_COMPOSED:
        dst = F32_COMPOSED[op](em, regs[0])
    elif op == "copy":
        dst = emit_reg(em, cls, f"{MOVES[cls]} {regs[0]}")
    elif cls == "f" and op == "relu":
        dst = emit_f32(em, f"max.f32 {regs[0]}, {f32_imm(0.0)}")
    elif op == "square":
        dst = emit_reg(em, cls, f"{BINOPS[cls]['multiply']} {regs[0]}, {regs[0]}")
    else:
        raise NotImplementedError(f"elementwise op {op!r} on {kind.name}")
    computed = F32 if kind is F16 else kind             # f16 arithmetic ran in f32, so it still needs rounding
    return emit_convert(em, dst, computed, result), result


def emit_assign(em: PtxEmitter, stmt: Assign) -> None:
    """Bind `stmt.name` to the op applied to its argument registers."""
    args = [(em.vars[a], em.type_of(a)) for a in stmt.args]
    result = ty(stmt.dtype) if stmt.dtype is not None else None
    reg, t = apply_op(em, stmt.op.name, args, result)
    em.bind(stmt.name, reg, t)


def emit_let(em: PtxEmitter, stmt: Let) -> None:
    """Bind a literal to a register, or an integer expression to its operand."""
    t = ty(stmt.dtype)
    if isinstance(stmt.value, Literal) and t.cls in ("f", "d"):
        dst = em.reg(t.cls)
        em.emit(f"mov.{t.ptx} {dst}, {imm(t, stmt.value.value)};")
        em.bind(stmt.name, dst, t)
    elif t is I32:
        em.bind(stmt.name, emit_int_expr(em, stmt.value), t)
    else:
        raise NotImplementedError(f"let {stmt.name} = {stmt.value!r} as {t.name}")


def emit_select(em: PtxEmitter, stmt: Select) -> None:
    """The first branch whose predicate holds; the last branch is the catch-all."""
    kind = em.type_of(stmt.branches[-1].value)
    dst = em.reg(kind.cls)
    em.emit(f"{MOVES[kind.cls]} {dst}, {em.vars[stmt.branches[-1].value]};")
    for branch in reversed(stmt.branches[:-1]):
        em.emit(f"{SELECTS[kind.cls]} {dst}, {em.vars[branch.value]}, {dst}, {emit_pred(em, branch.select)};")
    em.bind(stmt.name, dst, kind)
