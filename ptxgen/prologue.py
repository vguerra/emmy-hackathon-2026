"""The entry sequence, derived from the kernel's Tile."""

import math

from emmy.compiler.ir.kernel.ir import Smem, Tile

from .dtypes import ty
from .emitter import STATIC_SMEM_LIMIT, PtxEmitter
from .tma import TMA_TARGET, descriptor_names


def emit_prologue(em: PtxEmitter, tile: Tile, arg_order: list[str]) -> None:
    """Params -> global pointers, flat thread id, `gid < N` guard, axis decode."""
    if tile.aux_threads:
        raise NotImplementedError("Tile.aux_threads (warp-specialized producer band)")

    # Static shared memory is capped at 48 KiB; beyond that it must be one dynamic array.
    total = sum(ty(s.dtype).nbytes * math.prod(s.extents) for s in tile.body.iter_of_type(Smem))
    em.dynamic_smem = total > STATIC_SMEM_LIMIT

    for name in descriptor_names(tile):                 # tensor maps stay in param space, 128 bytes each
        em.tma_params[name] = f"param_{name}"
        em.target = TMA_TARGET
    for name in arg_order:
        if name in em.tma_params:
            continue
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
