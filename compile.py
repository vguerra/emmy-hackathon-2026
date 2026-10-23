import argparse
import json
import math
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write


def load_ir(ir_path: str) -> tuple[Graph, dict]:
    """Load the IR graph plus the meta.json (launch contract, arg order) next to it."""
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))
    with open(Path(ir_path).parent / "meta.json") as f:
        meta = json.load(f)
    return graph, meta


def find_kernel(graph: Graph) -> KernelOp:
    kernels = [graph.nodes[nid].op for nid in graph.topological_order()
               if isinstance(graph.nodes[nid].op, KernelOp)]
    if len(kernels) != 1:
        raise ValueError(f"expected exactly one KernelOp, found {len(kernels)}")
    return kernels[0]


def find_kernel_meta(meta: dict, entry: str) -> dict:
    for k in meta["kernels"]:
        if k["entry"] == entry:
            return k
    raise ValueError(f"no kernel named {entry!r} in meta.json")


def get_arg_order(meta: dict, entry: str) -> list[str]:
    """Kernel parameter names, in order, for the meta.json entry named `entry`."""
    return list(find_kernel_meta(meta, entry)["arg_order"])


def get_arg_types(meta: dict, entry: str) -> dict[str, tuple[str, list[int]]]:
    """Map each argument name to its (dtype, shape), e.g. 'x' -> ('f32', [4096])."""
    return {a["name"]: (a["dtype"], list(a["shape"]))
            for a in find_kernel_meta(meta, entry)["args"]}


def dump_graph(graph: Graph) -> None:
    """Pretty-print the graph to stderr (stdout is reserved for PTX)."""
    print(graph.pretty_print(), file=sys.stderr)

# Load


class PtxEmitter:
    """Collects body lines and hands out fresh virtual registers; the .reg
    declarations are only known once the walk is done, so render() adds them."""

    def __init__(self):
        self.lines = []
        self.shared = []                                # .shared declarations
        self.reg_counts = {"p": 0, "f": 0, "r": 0, "rd": 0}
        self.buffers = {}                               # IR buffer name -> global pointer reg
        self.vars = {}                                  # IR axis / SSA name -> register

    def reg(self, kind: str) -> str:
        self.reg_counts[kind] += 1
        return f"%{kind}{self.reg_counts[kind]}"

    def emit(self, line: str) -> None:
        self.lines.append("        " + line)

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")


def find_tile(kernel: KernelOp) -> Tile:
    tiles = kernel.body.of_type(Tile)
    if len(tiles) != 1:
        raise ValueError(f"expected exactly one top-level Tile, found {len(tiles)}")
    return tiles[0]


# Prologue

def emit_prologue(em: PtxEmitter, tile: Tile, arg_order: list[str]) -> None:
    """Params -> global pointers, flat thread id, `gid < N` guard, axis decode."""
    if tile.aux_threads:
        raise NotImplementedError("Tile.aux_threads (warp-specialized producer band)")

    for name in arg_order:
        raw, glob = em.reg("rd"), em.reg("rd")
        em.emit(f"ld.param.u64 {raw}, [param_{name}];")
        em.emit(f"cvta.to.global.u64 {glob}, {raw};")
        em.buffers[name] = glob

    # gid = blockIdx.x * blockDim.x + threadIdx.x
    tid, ctaid, ntid, gid = (em.reg("r") for _ in range(4))
    em.emit(f"mov.u32 {tid}, %tid.x;")
    em.emit(f"mov.u32 {ctaid}, %ctaid.x;")
    em.emit(f"mov.u32 {ntid}, %ntid.x;")
    em.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid};")

    # The launch may carry more threads than cells; this guard is not in the IR.
    guard = em.reg("p")
    em.emit(f"setp.ge.s32 {guard}, {gid}, {tile.n_elements};")
    em.emit(f"@{guard} bra DONE;")

    # axis = gid / stride % extent, stride = product of the extents to its right
    stride = 1
    decoded = []
    for axis, extent in zip(reversed(tile.axes), reversed(tile.extents)):
        decoded.append((axis.name, stride, extent))
        stride *= extent
    for i, (name, stride, extent) in enumerate(reversed(decoded)):
        reg = gid
        if stride > 1:
            reg = em.reg("r")
            em.emit(f"div.s32 {reg}, {gid}, {stride};")
        if i > 0:                                       # outermost axis needs no wrap
            wrapped = em.reg("r")
            em.emit(f"rem.s32 {wrapped}, {reg}, {extent};")
            reg = wrapped
        em.vars[name] = reg


# Walk

INT_BINOPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32",
              "/": "div.s32", "//": "div.s32", "%": "rem.s32"}

# .rn pins the rounding mode, which also stops ptxas from fusing mul+add into fma
# (the expected outputs come from numpy, which rounds every operation).
F32_BINOPS = {"add": "add.rn.f32", "subtract": "sub.rn.f32", "multiply": "mul.rn.f32",
              "divide": "div.rn.f32", "maximum": "max.f32", "minimum": "min.f32"}
F32_UNOPS = {"sqrt": "sqrt.rn.f32", "reciprocal": "rcp.rn.f32"}


def f32_imm(value: float) -> str:
    return "0f%08X" % struct.unpack(">I", struct.pack(">f", value))[0]


def is_imm(operand: str) -> bool:
    return operand.lstrip("-").isdigit()


def emit_int_expr(em: PtxEmitter, expr: Expr) -> str:
    """Compile an integer index expression; returns a register or an immediate."""
    if isinstance(expr, Var):
        return em.vars[expr.name]
    if isinstance(expr, Literal):
        return str(int(expr.value))
    if isinstance(expr, BinaryExpr) and expr.op in INT_BINOPS:
        left = emit_int_expr(em, expr.left)
        right = emit_int_expr(em, expr.right)
        dst = em.reg("r")
        em.emit(f"{INT_BINOPS[expr.op]} {dst}, {left}, {right};")
        return dst
    raise NotImplementedError(f"index expression {expr!r}")


def emit_flat_offset(em: PtxEmitter, index: tuple, shape: list[int]) -> str:
    """Row-major element offset of a coordinate tuple in a buffer of `shape`."""
    if not index:
        return "0"
    if len(index) != len(shape):
        raise NotImplementedError(f"index rank {len(index)} on a buffer of shape {shape}")
    offset = None
    stride = 1
    for expr, extent in zip(reversed(index), reversed(shape)):
        term = emit_int_expr(em, expr)
        if stride != 1:
            scaled = em.reg("r")
            em.emit(f"mul.lo.s32 {scaled}, {term}, {stride};")
            term = scaled
        if offset is None:
            offset = term
        else:
            total = em.reg("r")
            em.emit(f"add.s32 {total}, {offset}, {term};")
            offset = total
        stride *= extent
    return offset


def emit_address(em: PtxEmitter, buffer: str, offset: str, elem_bytes: int) -> str:
    addr = em.reg("rd")
    if is_imm(offset):
        em.emit(f"add.s64 {addr}, {em.buffers[buffer]}, {int(offset) * elem_bytes};")
    else:
        wide = em.reg("rd")
        em.emit(f"mul.wide.s32 {wide}, {offset}, {elem_bytes};")
        em.emit(f"add.s64 {addr}, {em.buffers[buffer]}, {wide};")
    return addr


def vector_suffix(n: int) -> str:
    if n not in (1, 2, 4):
        raise NotImplementedError(f"{n}-wide vector access")
    return "" if n == 1 else f".v{n}"


def emit_load(em: PtxEmitter, stmt: Load, arg_types: dict) -> None:
    if stmt.dtype.name != "f32":
        raise NotImplementedError(f"load dtype {stmt.dtype.name}")
    shape = arg_types[stmt.input][1]
    addr = emit_address(em, stmt.input, emit_flat_offset(em, stmt.index, shape), 4)
    regs = [em.reg("f") for _ in stmt.names]
    for name, reg in zip(stmt.names, regs):
        em.vars[name] = reg
    em.emit(f"ld.global{vector_suffix(len(regs))}.f32 {{{', '.join(regs)}}}, [{addr}];"
            if len(regs) > 1 else f"ld.global.f32 {regs[0]}, [{addr}];")


def emit_assign(em: PtxEmitter, stmt: Assign) -> None:
    op = stmt.op.name
    args = [em.vars[a] for a in stmt.args]
    dst = em.reg("f")
    if op in F32_BINOPS:
        em.emit(f"{F32_BINOPS[op]} {dst}, {args[0]}, {args[1]};")
    elif op in F32_UNOPS:
        em.emit(f"{F32_UNOPS[op]} {dst}, {args[0]};")
    elif op == "relu":
        em.emit(f"max.f32 {dst}, {args[0]}, {f32_imm(0.0)};")
    elif op == "exp":                                   # e^x = 2^(x * log2 e)
        scaled = em.reg("f")
        em.emit(f"mul.f32 {scaled}, {args[0]}, {f32_imm(math.log2(math.e))};")
        em.emit(f"ex2.approx.f32 {dst}, {scaled};")
    else:
        raise NotImplementedError(f"elementwise op {op!r}")
    em.vars[stmt.name] = dst


def emit_let(em: PtxEmitter, stmt: Let) -> None:
    if isinstance(stmt.value, Literal) and stmt.dtype.name == "f32":
        dst = em.reg("f")
        em.emit(f"mov.f32 {dst}, {f32_imm(float(stmt.value.value))};")
        em.vars[stmt.name] = dst
    elif stmt.dtype.name in ("i32", "int32"):
        em.vars[stmt.name] = emit_int_expr(em, stmt.value)
    else:
        raise NotImplementedError(f"let {stmt.name} = {stmt.value!r}")


def emit_write(em: PtxEmitter, stmt: Write, arg_types: dict) -> None:
    shape = arg_types[stmt.output][1]
    addr = emit_address(em, stmt.output, emit_flat_offset(em, stmt.index, shape), 4)
    values = [em.vars[v] for v in stmt.values]
    if stmt.atomic:                                     # no vector red on sm_89: one per element
        for i, value in enumerate(values):
            em.emit(f"red.global.add.f32 [{addr}+{4 * i}], {value};")
    elif len(values) > 1:
        em.emit(f"st.global{vector_suffix(len(values))}.f32 [{addr}], {{{', '.join(values)}}};")
    else:
        em.emit(f"st.global.f32 [{addr}], {values[0]};")


def emit_body(em: PtxEmitter, tile: Tile, arg_types: dict) -> None:
    """Visit tile.body statement by statement."""
    for stmt in tile.body:
        if isinstance(stmt, Load):
            emit_load(em, stmt, arg_types)
        elif isinstance(stmt, Assign):
            emit_assign(em, stmt)
        elif isinstance(stmt, Let):
            emit_let(em, stmt)
        elif isinstance(stmt, Write):
            emit_write(em, stmt, arg_types)
        else:
            raise NotImplementedError(f"statement {type(stmt).__name__}")


# Emit

def render(entry: str, arg_order: list[str], em: PtxEmitter) -> str:
    params = ",\n".join(f"        .param .u64 param_{name}" for name in arg_order)
    regs = [f"        .reg .{ty} %{kind}<{em.reg_counts[kind] + 1}>;"
            for kind, ty in (("p", "pred"), ("f", "f32"), ("r", "b32"), ("rd", "b64"))]
    return "\n".join([
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
        *em.shared,
        f".visible .entry {entry}(",
        params,
        ")",
        "{",
        *regs,
        *em.lines,
        "DONE:",
        "        ret;",
        "}",
        "",
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ir_json", help="path to the Kernel IR json file")
    parser.add_argument("--dump-graph", action="store_true",
                        help="pretty-print the graph to stderr")
    args = parser.parse_args()
    graph, meta = load_ir(args.ir_json)
    kernel = find_kernel(graph)
    arg_order = get_arg_order(meta, kernel.name)
    arg_types = get_arg_types(meta, kernel.name)
    if args.dump_graph:
        dump_graph(graph)
    tile = find_tile(kernel)
    em = PtxEmitter()
    emit_prologue(em, tile, arg_order)
    emit_body(em, tile, arg_types)
    print(render(kernel.name, arg_order, em))
