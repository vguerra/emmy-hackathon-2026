"""Control flow and cross-thread statements, and the dispatcher over every statement kind."""

from emmy.compiler.ir.expr import Expr, Literal
from emmy.compiler.ir.kernel.ir import (CpAsyncCommit, CpAsyncCopy, CpAsyncWait, FragmentApply, FragmentRepack,
                                        FragmentRowReduce, LdmatrixLoad, MbarrierArrive, MbarrierArriveExpectTx,
                                        MbarrierInit, MbarrierWait, MmaSyncPtx, RegFragment, RegStore, Smem, Sync,
                                        Tile, TmaDescriptor, TmaLoad, TreeHalve, WarpShuffle)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Select, Write

from .dtypes import emit_reg, emit_shfl_xor, identity_imm, ty
from .elementwise import BINOPS, MOVES, emit_assign, emit_let, emit_select, operand
from .emitter import PtxEmitter
from .expr import emit_int_expr, emit_pred
from .tma import (emit_mbarrier_arrive, emit_mbarrier_arrive_expect_tx, emit_mbarrier_init, emit_mbarrier_wait,
                  emit_tma_descriptor, emit_tma_load)
from .tensorcore import (emit_fragment_apply, emit_fragment_repack, emit_fragment_row_reduce, emit_ldmatrix,
                         emit_mma, emit_reg_fragment, emit_reg_store)
from .memory import emit_address, emit_cp_async, emit_ld, emit_load, emit_smem, emit_st, emit_write


def emit_cond(em: PtxEmitter, stmt: Cond, arg_types: dict) -> None:
    """if (cond) body else else_body, as a predicated branch around each side."""
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


def static_range(stmt: Loop | StridedLoop) -> range | None:
    """The iteration values of a loop whose start, end and step are all literals, else None."""
    end = getattr(stmt, "end", None)
    parts = (getattr(stmt, "start", Literal(0, "int")), end if end is not None else axis_end(stmt),
             getattr(stmt, "step", Literal(1, "int")))
    if all(isinstance(p, Literal) for p in parts):
        return range(*(int(p.value) for p in parts))
    return None


def indexes_fragments(stmt: Loop | StridedLoop) -> bool:
    """True when the body names a fragment register by this loop's axis, e.g. `_c0_[_r0]`.
    Registers cannot be indexed dynamically, so such a loop must be unrolled."""
    tag = f"[{stmt.axis.name}]"
    refs = []
    for s in stmt.body.iter():
        for field in ("frag", "c_frag", "a_frag", "b_frag", "pair_frag", "out"):
            refs.append(getattr(s, field, None))
        refs.extend(a for a in getattr(s, "args", ()) if isinstance(a, str))       # FragmentApply operands
        refs.extend(getattr(s, "frags", ()))                                       # FragmentRowReduce
    return any(isinstance(r, str) and tag in r for r in refs)


def emit_unrolled_loop(em: PtxEmitter, stmt: Loop | StridedLoop, values: range, arg_types: dict) -> None:
    """Emit the body once per iteration with the axis bound to an immediate."""
    for value in values:
        em.vars[stmt.axis.name] = str(value)
        emit_stmts(em, stmt.body, arg_types)


def emit_counted_loop(em: PtxEmitter, stmt: Loop | StridedLoop, start: Expr, end: Expr,
                      step: Expr, arg_types: dict) -> None:
    """for (axis = start; axis < end; axis += step) body, seeding the folded accumulators first."""
    values = static_range(stmt)
    if values is not None and indexes_fragments(stmt):
        return emit_unrolled_loop(em, stmt, values, arg_types)
    if stmt.seed:
        for acc in stmt.body.iter_of_type(Accum):
            t = ty(acc.dtype)
            if acc.name not in em.vars:                 # an Init may have declared it already
                em.bind(acc.name, em.reg(t.cls), t)
            em.emit(f"{MOVES[t.cls]} {em.vars[acc.name]}, {identity_imm(t, acc.op.name, acc.op.identity)};")

    index = em.reg("r")
    em.emit(f"mov.s32 {index}, {emit_int_expr(em, start)};")
    bound = emit_int_expr(em, end)
    step = emit_int_expr(em, step)
    top, done = em.new_label("LOOP"), em.new_label("LOOP_END")

    em.label(top)
    pred = em.reg("p")
    em.emit(f"setp.ge.s32 {pred}, {index}, {bound};")
    em.emit(f"@{pred} bra {done};")
    em.vars[stmt.axis.name] = index
    emit_stmts(em, stmt.body, arg_types)
    em.emit(f"add.s32 {index}, {index}, {step};")
    em.emit(f"bra {top};")
    em.label(done)


def axis_end(stmt: Loop | StridedLoop) -> Expr:
    """The loop bound when the IR leaves it implicit: the axis extent."""
    return Literal(stmt.axis.extent.as_static(), "int")


def emit_strided_loop(em: PtxEmitter, stmt: StridedLoop, arg_types: dict) -> None:
    """Each thread strides an axis from its own start by `step`."""
    end = stmt.end if stmt.end is not None else axis_end(stmt)
    emit_counted_loop(em, stmt, stmt.start, end, stmt.step, arg_types)


def emit_loop(em: PtxEmitter, stmt: Loop, arg_types: dict) -> None:
    """A plain counted loop over the whole axis."""
    emit_counted_loop(em, stmt, Literal(0, "int"), axis_end(stmt), Literal(1, "int"), arg_types)


def emit_init(em: PtxEmitter, stmt: Init) -> None:
    """Declare carried state seeded with a literal."""
    t = ty(stmt.dtype)
    em.bind(stmt.name, em.reg(t.cls), t)
    em.emit(f"{MOVES[t.cls]} {em.vars[stmt.name]}, {identity_imm(t, '', stmt.identity)};")


def emit_accum(em: PtxEmitter, stmt: Accum) -> None:
    """Fold `value` into the accumulator register, in place."""
    acc = em.vars[stmt.name]                            # in place: the register outlives the loop body
    left = em.vars[stmt.base] if stmt.base else acc     # base: a rescaled value read instead of acc
    t = em.type_of(stmt.name)
    value = operand(em, stmt.value, t)
    em.emit(f"{BINOPS[t.cls][stmt.op.name]} {acc}, {left}, {value};")


def emit_warp_shuffle(em: PtxEmitter, stmt: WarpShuffle) -> None:
    """Butterfly over `length` lanes: fold in the value of lane ^ offset, offset = length/2 .. 1."""
    offset = stmt.length // 2
    while offset >= 1:
        for state, other in zip(stmt.state, stmt.state_b):
            t = em.type_of(state)
            if t.cls not in ("f", "r"):
                raise NotImplementedError(f"warp shuffle of {t.name}")
            partner = emit_reg(em, t.cls, f"mov.b32 {emit_shfl_xor(em, em.vars[state], offset)}")
            em.bind(other, partner, t)
        for combine in stmt.combine_states:
            emit_assign(em, combine)
        offset //= 2


def emit_sync(em: PtxEmitter, stmt: Sync) -> None:
    """Block barrier, warp barrier, or a named barrier over `count` threads."""
    if stmt.warp:
        em.emit("bar.warp.sync 0xffffffff;")
    elif stmt.barrier_id == 0:
        em.emit("bar.sync 0;")
    else:
        em.emit(f"bar.sync {stmt.barrier_id}, {stmt.count};")


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
            t = em.smem_types[buf]
            for name, index in ((state, tid), (other, partner)):
                addr = emit_address(em, buf, index, t.nbytes)
                em.bind(name, emit_ld(em, ".shared", addr, t, 1)[0], t)
        for combine in stmt.combine_states:
            emit_assign(em, combine)
        for buf, state in zip(stmt.bufs, stmt.state):
            t = em.smem_types[buf]
            emit_st(em, ".shared", emit_address(em, buf, tid, t.nbytes), t, [em.vars[state]])
        em.label(skip)
        emit_sync(em, Sync(stmt.barrier_id, stmt.barrier_count))
        stride //= 2
    for buf, state in zip(stmt.bufs, stmt.state):       # broadcast the total
        t = em.smem_types[buf]
        em.bind(state, emit_ld(em, ".shared", emit_address(em, buf, "0", t.nbytes), t, 1)[0], t)


def emit_stmts(em: PtxEmitter, body, arg_types: dict) -> None:
    """Lower each statement of a body in order, dispatching on its kind."""
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
        elif isinstance(stmt, TmaDescriptor):
            emit_tma_descriptor(em, stmt)
        elif isinstance(stmt, TmaLoad):
            emit_tma_load(em, stmt, arg_types)
        elif isinstance(stmt, MbarrierInit):
            emit_mbarrier_init(em, stmt, arg_types)
        elif isinstance(stmt, MbarrierArriveExpectTx):
            emit_mbarrier_arrive_expect_tx(em, stmt, arg_types)
        elif isinstance(stmt, MbarrierArrive):
            emit_mbarrier_arrive(em, stmt, arg_types)
        elif isinstance(stmt, MbarrierWait):
            emit_mbarrier_wait(em, stmt, arg_types)
        elif isinstance(stmt, RegFragment):
            emit_reg_fragment(em, stmt)
        elif isinstance(stmt, LdmatrixLoad):
            emit_ldmatrix(em, stmt, arg_types)
        elif isinstance(stmt, MmaSyncPtx):
            emit_mma(em, stmt)
        elif isinstance(stmt, FragmentApply):
            emit_fragment_apply(em, stmt, arg_types)
        elif isinstance(stmt, FragmentRowReduce):
            emit_fragment_row_reduce(em, stmt)
        elif isinstance(stmt, FragmentRepack):
            emit_fragment_repack(em, stmt)
        elif isinstance(stmt, RegStore):
            emit_reg_store(em, stmt, arg_types)
        elif isinstance(stmt, Select):
            emit_select(em, stmt)
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
    """Lower everything inside the Tile."""
    emit_stmts(em, tile.body, arg_types)
