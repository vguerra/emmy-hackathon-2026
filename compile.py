"""Kernel IR -> PTX.

    python compile.py path/to/<kernel>.ir.json > out.ptx

Reads the IR with emmy's Graph.from_dict and the meta.json beside it, then lowers the
kernel itself (see the ptxgen package). stdout carries PTX only; --dump-graph prints to stderr.
"""

import argparse

from ptxgen.control import emit_body
from ptxgen.emitter import PtxEmitter, render
from ptxgen.loader import (dump_graph, find_kernel, find_tile, get_arg_order, get_arg_types,
                           load_ir)
from ptxgen.prologue import emit_prologue


def compile_ir(ir_path: str, dump: bool = False) -> str:
    """PTX text for the kernel in the IR file at `ir_path`."""
    graph, meta = load_ir(ir_path)
    kernel = find_kernel(graph)
    arg_order = get_arg_order(meta, kernel.name)
    arg_types = get_arg_types(meta, kernel.name)
    if dump:
        dump_graph(graph)
    tile = find_tile(kernel)
    em = PtxEmitter()
    em.allow_fma = meta.get("rtol", 0) > 0 or meta.get("atol", 0) > 0
    emit_prologue(em, tile, arg_order)
    emit_body(em, tile, arg_types)
    return render(kernel.name, arg_order, em)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ir_json", help="path to the Kernel IR json file")
    parser.add_argument("--dump-graph", action="store_true",
                        help="pretty-print the graph to stderr")
    args = parser.parse_args()
    print(compile_ir(args.ir_json, args.dump_graph))


if __name__ == "__main__":
    main()
