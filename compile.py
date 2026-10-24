import argparse
import json
import math
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Smem,
                                        Sync, Tile, TreeHalve, WarpShuffle)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write


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
        self.smem_base = {}                             # shared array name -> u32 base address reg
        self.smem_shapes = {}                           # shared array name -> extents
        self.label_count = 0
        self.dynamic_smem = False                       # shared arrays live in one extern array
        self.smem_used = 0                              # bytes handed out of it so far

    def reg(self, kind: str) -> str:
        self.reg_counts[kind] += 1
        return f"%{kind}{self.reg_counts[kind]}"

    def emit(self, line: str) -> None:
        self.lines.append("        " + line)

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def new_label(self, prefix: str) -> str:
        self.label_count += 1
        return f"{prefix}_{self.label_count}"


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

    # Static shared memory is capped at 48 KiB; beyond that it must be one dynamic array.
    total = sum(4 * math.prod(s.extents) for s in tile.body.iter_of_type(Smem))
    em.dynamic_smem = total > STATIC_SMEM_LIMIT

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

    # Cooperative tiles shuffle and barrier below the guard, so no thread may be cut off.
    if tile.block_threads and tile.n_elements % tile.block_threads:
        raise NotImplementedError("cooperative Tile whose N is not a multiple of block_threads")

    # lane / warp are free variables of the reduction statements
    em.vars["lane"], em.vars["warp"] = em.reg("r"), em.reg("r")
    em.emit(f"and.b32 {em.vars['lane']}, {tid}, 31;")
    em.emit(f"shr.u32 {em.vars['warp']}, {tid}, 5;")

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

STATIC_SMEM_LIMIT = 48 * 1024

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
    if isinstance(expr, TernaryExpr):
        pred = emit_pred(em, expr.cond)
        if_true, if_false = emit_int_expr(em, expr.if_true), emit_int_expr(em, expr.if_false)
        dst = em.reg("r")
        em.emit(f"selp.s32 {dst}, {if_true}, {if_false}, {pred};")
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


def buffer_shape(em: PtxEmitter, arg_types: dict, buffer: str) -> list[int]:
    return list(em.smem_shapes[buffer]) if buffer in em.smem_base else arg_types[buffer][1]


def mem_space(em: PtxEmitter, buffer: str) -> str:
    return ".shared" if buffer in em.smem_base else ".global"


def emit_address(em: PtxEmitter, buffer: str, offset: str, elem_bytes: int) -> str:
    if buffer in em.smem_base:                          # shared addresses are 32-bit
        addr = em.reg("r")
        if is_imm(offset):
            em.emit(f"add.s32 {addr}, {em.smem_base[buffer]}, {int(offset) * elem_bytes};")
        else:
            scaled = em.reg("r")
            em.emit(f"mul.lo.s32 {scaled}, {offset}, {elem_bytes};")
            em.emit(f"add.s32 {addr}, {em.smem_base[buffer]}, {scaled};")
        return addr
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
    if stmt.dtype is not None and stmt.dtype.name != "f32":   # None: untyped (shared-memory) load
        raise NotImplementedError(f"load dtype {stmt.dtype.name}")
    shape = buffer_shape(em, arg_types, stmt.input)
    addr = emit_address(em, stmt.input, emit_flat_offset(em, stmt.index, shape), 4)
    regs = [em.reg("f") for _ in stmt.names]
    for name, reg in zip(stmt.names, regs):
        em.vars[name] = reg
    space = mem_space(em, stmt.input)
    em.emit(f"ld{space}{vector_suffix(len(regs))}.f32 {{{', '.join(regs)}}}, [{addr}];"
            if len(regs) > 1 else f"ld{space}.f32 {regs[0]}, [{addr}];")


def emit_assign(em: PtxEmitter, stmt: Assign) -> None:
    op = stmt.op.name
    args = [em.vars[a] for a in stmt.args]
    dst = em.reg("f")
    if op in F32_BINOPS:
        em.emit(f"{F32_BINOPS[op]} {dst}, {args[0]}, {args[1]};")
    elif op in F32_UNOPS:
        em.emit(f"{F32_UNOPS[op]} {dst}, {args[0]};")
    elif op == "copy":
        em.emit(f"mov.f32 {dst}, {args[0]};")
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
    shape = buffer_shape(em, arg_types, stmt.output)
    addr = emit_address(em, stmt.output, emit_flat_offset(em, stmt.index, shape), 4)
    values = [em.vars[v] for v in stmt.values]
    space = mem_space(em, stmt.output)
    if stmt.atomic:                                     # no vector red on sm_89: one per element
        for i, value in enumerate(values):
            em.emit(f"red.global.add.f32 [{addr}+{4 * i}], {value};")
    elif len(values) > 1:
        em.emit(f"st{space}{vector_suffix(len(values))}.f32 [{addr}], {{{', '.join(values)}}};")
    else:
        em.emit(f"st{space}.f32 [{addr}], {values[0]};")


PRED_CMPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


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


def emit_cond(em: PtxEmitter, stmt: Cond, arg_types: dict) -> None:
    pred = emit_pred(em, stmt.cond)
    end = em.new_label("ENDIF")
    if not stmt.else_body:
        em.emit(f"@!{pred} bra {end};")
        emit_stmts(em, stmt.body, arg_types)
    else:
        other = em.new_label("ELSE")
        em.emit(f"@!{pred} bra {other};")
        emit_stmts(em, stmt.body, arg_types)
        em.emit(f"bra {end};")
        em.label(other)
        emit_stmts(em, stmt.else_body, arg_types)
    em.label(end)


def emit_counted_loop(em: PtxEmitter, stmt: Loop | StridedLoop, start: Expr, end: Expr,
                      step: Expr, arg_types: dict) -> None:
    """for (axis = start; axis < end; axis += step) body, seeding the folded accumulators first."""
    if stmt.seed:
        for acc in stmt.body.iter_of_type(Accum):
            if acc.name not in em.vars:                 # an Init may have declared it already
                em.vars[acc.name] = em.reg("f")
            em.emit(f"mov.f32 {em.vars[acc.name]}, {f32_imm(acc.op.identity)};")

    index = em.reg("r")
    em.emit(f"mov.s32 {index}, {emit_int_expr(em, start)};")
    bound = emit_int_expr(em, end)
    step = emit_int_expr(em, step)
    top, done = em.new_label("LOOP"), em.new_label("LOOP_END")
    em.vars[stmt.axis.name] = index

    em.label(top)
    pred = em.reg("p")
    em.emit(f"setp.ge.s32 {pred}, {index}, {bound};")
    em.emit(f"@{pred} bra {done};")
    emit_stmts(em, stmt.body, arg_types)
    em.emit(f"add.s32 {index}, {index}, {step};")
    em.emit(f"bra {top};")
    em.label(done)


def axis_end(stmt: Loop | StridedLoop) -> Expr:
    return Literal(stmt.axis.extent.as_static(), "int")


def emit_strided_loop(em: PtxEmitter, stmt: StridedLoop, arg_types: dict) -> None:
    end = stmt.end if stmt.end is not None else axis_end(stmt)
    emit_counted_loop(em, stmt, stmt.start, end, stmt.step, arg_types)


def emit_loop(em: PtxEmitter, stmt: Loop, arg_types: dict) -> None:
    emit_counted_loop(em, stmt, Literal(0, "int"), axis_end(stmt), Literal(1, "int"), arg_types)


def emit_init(em: PtxEmitter, stmt: Init) -> None:
    if stmt.dtype.name != "f32":
        raise NotImplementedError(f"init dtype {stmt.dtype.name}")
    em.vars[stmt.name] = em.reg("f")
    em.emit(f"mov.f32 {em.vars[stmt.name]}, {f32_imm(stmt.identity)};")


def emit_cp_async(em: PtxEmitter, stmt: CpAsyncCopy, arg_types: dict) -> None:
    """Global -> shared copy of 4/8/16 bytes; 16-byte copies must use .cg."""
    if stmt.swizzle != "NONE":
        raise NotImplementedError("CpAsyncCopy swizzle")
    if stmt.nbytes not in (4, 8, 16):
        raise NotImplementedError(f"cp.async of {stmt.nbytes} bytes")
    smem_shape = buffer_shape(em, arg_types, stmt.smem)
    offset = emit_flat_offset(em, stmt.smem_index, smem_shape)
    if stmt.lane_index is not None:                     # this thread's own (row, col) within the stripe
        row, col = (emit_int_expr(em, e) for e in stmt.lane_index)
        lane = em.reg("r")
        em.emit(f"mad.lo.s32 {lane}, {row}, {smem_shape[-1]}, {col};")
        total = em.reg("r")
        em.emit(f"add.s32 {total}, {offset}, {lane};")
        offset = total
    dst = emit_address(em, stmt.smem, offset, 4)
    src = emit_address(em, stmt.src, emit_flat_offset(em, stmt.src_index, buffer_shape(em, arg_types, stmt.src)), 4)
    cache = "cg" if stmt.nbytes == 16 else "ca"
    em.emit(f"cp.async.{cache}.shared.global [{dst}], [{src}], {stmt.nbytes};")


def emit_accum(em: PtxEmitter, stmt: Accum) -> None:
    acc = em.vars[stmt.name]                            # in place: the register outlives the loop body
    left = em.vars[stmt.base] if stmt.base else acc     # base: a rescaled value read instead of acc
    em.emit(f"{F32_BINOPS[stmt.op.name]} {acc}, {left}, {em.vars[stmt.value]};")


def emit_warp_shuffle(em: PtxEmitter, stmt: WarpShuffle) -> None:
    """Butterfly over `length` lanes: fold in the value of lane ^ offset, offset = length/2 .. 1."""
    offset = stmt.length // 2
    while offset >= 1:
        for state, other in zip(stmt.state, stmt.state_b):
            bits, shuffled, partner = em.reg("r"), em.reg("r"), em.reg("f")
            em.emit(f"mov.b32 {bits}, {em.vars[state]};")
            em.emit(f"shfl.sync.bfly.b32 {shuffled}, {bits}, {offset}, 0x1f, 0xffffffff;")
            em.emit(f"mov.b32 {partner}, {shuffled};")
            em.vars[other] = partner
        for combine in stmt.combine_states:
            emit_assign(em, combine)
        offset //= 2


def emit_sync(em: PtxEmitter, stmt: Sync) -> None:
    if stmt.warp:
        em.emit("bar.warp.sync 0xffffffff;")
    elif stmt.barrier_id == 0:
        em.emit("bar.sync 0;")
    else:
        em.emit(f"bar.sync {stmt.barrier_id}, {stmt.count};")


def emit_smem(em: PtxEmitter, stmt: Smem) -> None:
    if stmt.dtype != "float":
        raise NotImplementedError(f"shared dtype {stmt.dtype}")
    align = max(stmt.align, 4)
    nbytes = 4 * math.prod(stmt.extents)
    em.smem_base[stmt.name] = em.reg("r")
    em.smem_shapes[stmt.name] = stmt.extents
    if em.dynamic_smem:                                 # carve the array out of the extern one
        em.smem_used = -(-em.smem_used // align) * align
        em.emit(f"mov.u32 {em.smem_base[stmt.name]}, smem_dyn;")
        if em.smem_used:
            em.emit(f"add.s32 {em.smem_base[stmt.name]}, {em.smem_base[stmt.name]}, {em.smem_used};")
        em.smem_used += nbytes
    else:
        em.shared.append(f".shared .align {align} .b8 {stmt.name}[{nbytes}];")
        em.emit(f"mov.u32 {em.smem_base[stmt.name]}, {stmt.name};")


def emit_tree_halve(em: PtxEmitter, stmt: TreeHalve, arg_types: dict) -> None:
    """Halving tree over the partials in shared memory; every thread ends with the total."""
    if stmt.inner is not None:
        raise NotImplementedError("TreeHalve.inner")
    tid = em.vars[stmt.tid_var]
    stride = stmt.length // 2
    while stride >= 1:
        skip = em.new_label("TREE_SKIP")
        active = em.reg("p")
        em.emit(f"setp.lt.s32 {active}, {tid}, {stride};")
        em.emit(f"@!{active} bra {skip};")
        partner = em.reg("r")
        em.emit(f"add.s32 {partner}, {tid}, {stride};")
        for buf, state, other in zip(stmt.bufs, stmt.state, stmt.state_b):
            for name, index in ((state, tid), (other, partner)):
                addr = emit_address(em, buf, index, 4)
                em.vars[name] = em.reg("f")
                em.emit(f"ld.shared.f32 {em.vars[name]}, [{addr}];")
        for combine in stmt.combine_states:
            emit_assign(em, combine)
        for buf, state in zip(stmt.bufs, stmt.state):
            em.emit(f"st.shared.f32 [{emit_address(em, buf, tid, 4)}], {em.vars[state]};")
        em.label(skip)
        emit_sync(em, Sync(stmt.barrier_id, stmt.barrier_count))
        stride //= 2
    for buf, state in zip(stmt.bufs, stmt.state):       # broadcast the total
        em.vars[state] = em.reg("f")
        em.emit(f"ld.shared.f32 {em.vars[state]}, [{emit_address(em, buf, '0', 4)}];")


def emit_stmts(em: PtxEmitter, body, arg_types: dict) -> None:
    for stmt in body:
        if isinstance(stmt, Load):
            emit_load(em, stmt, arg_types)
        elif isinstance(stmt, Assign):
            emit_assign(em, stmt)
        elif isinstance(stmt, Let):
            emit_let(em, stmt)
        elif isinstance(stmt, Write):
            emit_write(em, stmt, arg_types)
        elif isinstance(stmt, StridedLoop):
            emit_strided_loop(em, stmt, arg_types)
        elif isinstance(stmt, Loop):
            emit_loop(em, stmt, arg_types)
        elif isinstance(stmt, Init):
            emit_init(em, stmt)
        elif isinstance(stmt, CpAsyncCopy):
            emit_cp_async(em, stmt, arg_types)
        elif isinstance(stmt, CpAsyncCommit):
            em.emit("cp.async.commit_group;")
        elif isinstance(stmt, CpAsyncWait):
            em.emit(f"cp.async.wait_group {stmt.group};")
        elif isinstance(stmt, Accum):
            emit_accum(em, stmt)
        elif isinstance(stmt, WarpShuffle):
            emit_warp_shuffle(em, stmt)
        elif isinstance(stmt, Smem):
            emit_smem(em, stmt)
        elif isinstance(stmt, Sync):
            emit_sync(em, stmt)
        elif isinstance(stmt, TreeHalve):
            emit_tree_halve(em, stmt, arg_types)
        elif isinstance(stmt, Cond):
            emit_cond(em, stmt, arg_types)
        else:
            raise NotImplementedError(f"statement {type(stmt).__name__}")


def emit_body(em: PtxEmitter, tile: Tile, arg_types: dict) -> None:
    emit_stmts(em, tile.body, arg_types)


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
        *([".extern .shared .align 16 .b8 smem_dyn[];"] if em.dynamic_smem else []),
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
