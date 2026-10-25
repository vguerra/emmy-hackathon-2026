"""Reading the inputs: the IR graph, the meta.json beside it, and what we need from each."""

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp, Tile


def load_ir(ir_path: str) -> tuple[Graph, dict]:
    """Load the IR graph plus the meta.json (launch contract, arg order) next to it."""
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))
    with open(Path(ir_path).parent / "meta.json") as f:
        meta = json.load(f)
    return graph, meta


def find_kernel(graph: Graph) -> KernelOp:
    """The graph's single KernelOp; its body is the statement tree to lower."""
    kernels = [graph.nodes[nid].op for nid in graph.topological_order()
               if isinstance(graph.nodes[nid].op, KernelOp)]
    if len(kernels) != 1:
        raise ValueError(f"expected exactly one KernelOp, found {len(kernels)}")
    return kernels[0]


def find_tile(kernel: KernelOp) -> Tile:
    """The kernel's top-level Tile, which defines the thread-to-cell mapping."""
    tiles = kernel.body.of_type(Tile)
    if len(tiles) != 1:
        raise ValueError(f"expected exactly one top-level Tile, found {len(tiles)}")
    return tiles[0]


def find_kernel_meta(meta: dict, entry: str) -> dict:
    """The meta.json entry for the kernel named `entry`."""
    for k in meta["kernels"]:
        if k["entry"] == entry:
            return k
    raise ValueError(f"no kernel named {entry!r} in meta.json")


def get_arg_order(meta: dict, entry: str) -> list[str]:
    """Kernel parameter names, in order. meta.json, not the IR node, is the contract."""
    return list(find_kernel_meta(meta, entry)["arg_order"])


def get_arg_types(meta: dict, entry: str) -> dict[str, tuple[str, list[int]]]:
    """Map each argument name to its (dtype, shape), e.g. 'x' -> ('f32', [4096])."""
    return {a["name"]: (a["dtype"], list(a["shape"]))
            for a in find_kernel_meta(meta, entry)["args"]}


def dump_graph(graph: Graph) -> None:
    """Pretty-print the graph to stderr (stdout is reserved for PTX)."""
    print(graph.pretty_print(), file=sys.stderr)
