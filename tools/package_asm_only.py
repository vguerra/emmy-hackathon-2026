#!/usr/bin/env python3
"""Package an emmy dump into an assemble-only problem: the Kernel IR plus a meta.json, no expected output.

For kernels the runner cannot launch (a TMA kernel takes a host-built tensor map as an argument,
which the runner's buffer/scalar API cannot pass), so generate.py cannot package them either.

    EMMY_DUMP_DIR=out EMMY_KNOBS=STAGE=d2/smem-tma emmy compile -c "<torch expr>" --ir cuda --target sm_120
    python tools/package_asm_only.py out problems_asm/t_h_mm_tma
    python tools/check_assembles.py problems_asm
"""

import json
import sys
from pathlib import Path


def describe(name: str, tensors: dict) -> dict:
    """A buffer argument's dtype and shape; anything else is a host-built tensor map."""
    if name in tensors:
        return {"name": name, "dtype": tensors[name]["dtype"], "shape": tensors[name]["shape"] or [1]}
    return {"name": name, "dtype": "tensormap", "shape": [128]}


def main(dump: Path, out: Path) -> None:
    kernel_graph = json.loads(next(dump.glob("*_lowering_kernel.json")).read_text())
    cuda_graph = json.loads(next(dump.glob("*_lowering_cuda.json")).read_text())
    out.mkdir(parents=True, exist_ok=True)
    tensors = {t["name"]: t for n in kernel_graph["nodes"].values() for t in n.get("outputs", [])}
    kernels = []
    for node in cuda_graph["nodes"].values():
        if node["op"] != "CudaOp":
            continue
        fields = node["op_fields"]
        kernels.append({"entry": fields["kernel_name"], "arg_order": fields["arg_order"],
                        "grid": fields["grid"], "block": fields["block"], "smem_bytes": fields.get("smem_bytes", 0),
                        "args": [describe(name, tensors) for name in fields["arg_order"]]})
    (entry,) = [k["entry"] for k in kernels]
    for node in kernel_graph["nodes"].values():
        if node["op"] == "KernelOp":
            node["op_fields"]["name"] = entry
    (out / f"{entry}.ir.json").write_text(json.dumps(kernel_graph, indent=2))
    (out / "meta.json").write_text(json.dumps({"id": out.name, "assemble_only": True, "kernels": kernels}, indent=1))
    print(f"wrote {out}/{entry}.ir.json and meta.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
