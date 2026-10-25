"""Shared helpers: an emitter to lower into, a tiny interpreter for straight-line integer PTX, and PTX linting."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptxgen.emitter import PtxEmitter  # noqa: E402


def lowered(build) -> list[str]:
    """Run `build(em)` against a fresh emitter and return the emitted PTX lines, stripped."""
    em = PtxEmitter()
    build(em)
    return [line.strip() for line in em.lines]


def run_integer_ptx(lines: list[str], registers: dict) -> dict:
    """Execute straight-line shr/shl/and/xor/add/mul PTX over u32 registers; returns the register file."""
    ops = {
        "shr.u32": lambda a, b: a >> b, "shl.b32": lambda a, b: (a << b) & 0xFFFFFFFF,
        "and.b32": lambda a, b: a & b, "xor.b32": lambda a, b: a ^ b,
        "add.s32": lambda a, b: a + b, "mul.lo.s32": lambda a, b: a * b,
    }
    registers = dict(registers)

    def value(token: str) -> int:
        return registers[token] if token.startswith("%") else int(token)

    for line in lines:
        opcode, operands = line.rstrip(";").split(" ", 1)
        dst, *sources = [t.strip() for t in operands.split(",")]
        registers[dst] = ops[opcode](*(value(s) for s in sources))
    return registers


REG_DECL = re.compile(r"\.reg \.\w+\s+%(\w+?)<(\d+)>;")
REG_USE = re.compile(r"%(p|f|h|d|rd|r)(\d+)\b")
BRANCH = re.compile(r"\bbra\s+(\w+);")
LABEL = re.compile(r"^(\w+):", re.MULTILINE)


def lint_ptx(ptx: str, entry: str, arg_order: list[str]) -> list[str]:
    """Structural problems in generated PTX, found without ptxas: an empty list means none."""
    problems = []
    if not ptx.isascii():
        problems.append("non-ASCII characters (ptxas rejects them)")
    first = next(line for line in ptx.splitlines() if line.strip())
    if not first.startswith(".version"):
        problems.append(f"starts with {first!r}, not .version")
    if f".visible .entry {entry}(" not in ptx:
        problems.append(f"entry {entry!r} missing")
    if ptx.count("{") != ptx.count("}"):
        problems.append("unbalanced braces")
    params = re.findall(r"\.param [^\n]*?param_(\w+?)(?:\[\d+\])?[,\n]", ptx + "\n")
    if params != arg_order:
        problems.append(f"parameters {params} differ from arg_order {arg_order}")
    declared = {kind: int(count) for kind, count in REG_DECL.findall(ptx)}
    for kind, index in REG_USE.findall(ptx):
        if int(index) >= declared.get(kind, 0):
            problems.append(f"register %{kind}{index} used but not declared")
            break
    defined = set(LABEL.findall(ptx))
    for target in BRANCH.findall(ptx):
        if target not in defined:
            problems.append(f"branch to undefined label {target}")
    return problems
