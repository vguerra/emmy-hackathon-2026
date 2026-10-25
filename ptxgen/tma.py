"""Tensor Memory Accelerator copies (cp.async.bulk.tensor) and the mbarriers that signal them.

A TMA copy is issued by one thread and completes asynchronously: the issuer announces the byte
count with `mbarrier.arrive.expect_tx`, the hardware decrements it as the box lands in shared
memory, and consumers wait on the barrier's phase parity.

The tensor map (descriptor) is built on the host with cuTensorMapEncodeTiled and arrives as a
128-byte, 64-byte-aligned kernel parameter. The runner cannot pass one, so kernels using these
statements are checked for assembly with ptxas (tools/check_assembles.py) but not executed.
"""

from emmy.compiler.ir.kernel.ir import (MbarrierArrive, MbarrierArriveExpectTx, MbarrierInit, MbarrierWait,
                                        TmaDescriptor, TmaLoad)

from .dtypes import emit_reg
from .emitter import PtxEmitter
from .expr import emit_flat_offset, emit_int_expr
from .memory import buffer_shape, buffer_type, emit_address

TMA_TARGET = "sm_90"        # cp.async.bulk.tensor and mbarrier.expect_tx need sm_90 or later
MBARRIER_BYTES = 8


def descriptor_names(tile) -> list[str]:
    """Names of the tensor-map kernel parameters a Tile's TmaDescriptor statements declare."""
    return [d.name for d in tile.body.iter_of_type(TmaDescriptor)]


def emit_tma_descriptor(em: PtxEmitter, stmt: TmaDescriptor) -> None:
    """Declarative: the descriptor is a kernel parameter built on the host, so nothing is emitted."""
    if stmt.name not in em.tma_params:
        raise ValueError(f"tensor map {stmt.name!r} is not a kernel parameter")


def barrier_address(em: PtxEmitter, arg_types: dict, name: str, slot) -> str:
    """32-bit shared address of an mbarrier, or of one slot of an mbarrier array."""
    offset = emit_int_expr(em, slot) if slot is not None else "0"
    return emit_address(em, name, offset, MBARRIER_BYTES)


def emit_mbarrier_init(em: PtxEmitter, stmt: MbarrierInit, arg_types: dict) -> None:
    em.emit(f"mbarrier.init.shared::cta.b64 [{barrier_address(em, arg_types, stmt.mbar, stmt.slot)}], {stmt.count};")


def emit_mbarrier_arrive_expect_tx(em: PtxEmitter, stmt: MbarrierArriveExpectTx, arg_types: dict) -> None:
    addr = barrier_address(em, arg_types, stmt.mbar, stmt.slot)
    em.emit(f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [{addr}], {stmt.bytes_};")


def emit_mbarrier_arrive(em: PtxEmitter, stmt: MbarrierArrive, arg_types: dict) -> None:
    em.emit(f"mbarrier.arrive.shared::cta.b64 _, [{barrier_address(em, arg_types, stmt.mbar, stmt.slot)}];")


def emit_mbarrier_wait(em: PtxEmitter, stmt: MbarrierWait, arg_types: dict) -> None:
    """Spin until the barrier's phase with the given parity has completed."""
    addr = barrier_address(em, arg_types, stmt.mbar, stmt.slot)
    phase = emit_int_expr(em, stmt.phase)
    spin = em.new_label("MBAR_WAIT")
    done = em.reg("p")
    em.label(spin)
    em.emit(f"mbarrier.try_wait.parity.shared::cta.b64 {done}, [{addr}], {phase};")
    em.emit(f"@!{done} bra {spin};")


def emit_tma_load(em: PtxEmitter, stmt: TmaLoad, arg_types: dict) -> None:
    """Copy one box into shared memory; completion is signalled on the mbarrier."""
    smem_type = buffer_type(em, arg_types, stmt.smem)
    dst = emit_address(em, stmt.smem, emit_flat_offset(em, stmt.smem_index, buffer_shape(em, arg_types, stmt.smem)),
                       smem_type.nbytes)
    barrier = barrier_address(em, arg_types, stmt.mbar, stmt.mbar_slot)
    # PTX coordinates run innermost-first, the reverse of the IR's C order
    coords = [emit_reg(em, "r", f"mov.s32 {emit_int_expr(em, c)}") for c in reversed(stmt.coords)]
    # a kernel parameter's generic address, as nvcc takes it for a __grid_constant__ tensor map
    param = emit_reg(em, "rd", f"mov.u64 {em.tma_params[stmt.desc]}")
    generic = emit_reg(em, "rd", f"cvta.param.u64 {param}")
    em.emit(f"cp.async.bulk.tensor.{len(coords)}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
            f"[{dst}], [{generic}, {{{', '.join(coords)}}}], [{barrier}];")
