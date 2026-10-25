"""Kernel IR -> PTX lowering.

loader       read the IR graph and meta.json, locate the kernel and its Tile
emitter      PtxEmitter (lines, registers, symbol tables) and the final text assembly
prologue     entry sequence: params, thread id, bounds guard, axis decode
expr         index and predicate expressions
memory       global and shared memory: addressing, load, store, cp.async, Smem
elementwise  scalar arithmetic: Assign, Let, Select
control      loops, conditions, accumulators, warp/block reductions, and the statement dispatcher
"""
