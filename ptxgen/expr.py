"""Index and predicate expressions (the IR's coordinate sublanguage)."""

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var

from .emitter import PtxEmitter

INT_BINOPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32",
              "/": "div.s32", "//": "div.s32", "%": "rem.s32"}
PRED_CMPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


def int_op(em: PtxEmitter, op: str, left: str, right: str) -> str:
    """`left op right` on s32 operands, one instruction."""
    dst = em.reg("r")
    em.emit(f"{INT_BINOPS[op]} {dst}, {left}, {right};")
    return dst


def emit_int_expr(em: PtxEmitter, expr: Expr) -> str:
    """Compile an integer index expression; returns a register or an immediate."""
    if isinstance(expr, Var):
        return em.vars[expr.name]
    if isinstance(expr, Literal):
        return str(int(expr.value))
    if isinstance(expr, BinaryExpr) and expr.op in INT_BINOPS:
        return int_op(em, expr.op, emit_int_expr(em, expr.left), emit_int_expr(em, expr.right))
    if isinstance(expr, TernaryExpr):
        pred = emit_pred(em, expr.cond)
        if_true, if_false = emit_int_expr(em, expr.if_true), emit_int_expr(em, expr.if_false)
        dst = em.reg("r")
        em.emit(f"selp.s32 {dst}, {if_true}, {if_false}, {pred};")
        return dst
    raise NotImplementedError(f"index expression {expr!r}")


def emit_pred(em: PtxEmitter, expr: Expr) -> str:
    """Compile a predicate expression into a predicate register."""
    if isinstance(expr, BinaryExpr) and expr.op in PRED_CMPS:
        left, right = emit_int_expr(em, expr.left), emit_int_expr(em, expr.right)
        pred = em.reg("p")
        em.emit(f"setp.{PRED_CMPS[expr.op]}.s32 {pred}, {left}, {right};")
        return pred
    if isinstance(expr, BinaryExpr) and expr.op in ("&&", "||"):
        left, right = emit_pred(em, expr.left), emit_pred(em, expr.right)
        pred = em.reg("p")
        em.emit(f"{'and' if expr.op == '&&' else 'or'}.pred {pred}, {left}, {right};")
        return pred
    raise NotImplementedError(f"predicate {expr!r}")


def emit_flat_offset(em: PtxEmitter, index: tuple, shape: list[int]) -> str:
    """Row-major element offset of a coordinate tuple in a buffer of `shape`."""
    if not index:
        return "0"
    if len(index) != len(shape):
        raise NotImplementedError(f"index rank {len(index)} on a buffer of shape {shape}")
    offset = None
    stride = 1
    for expr, extent in zip(reversed(index), reversed(shape)):
        term = int_op(em, "*", emit_int_expr(em, expr), str(stride))
        offset = term if offset is None else int_op(em, "+", offset, term)
        stride *= extent
    return offset
