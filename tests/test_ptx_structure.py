"""Compile every problem folder and lint the PTX without ptxas or a GPU.

The lint catches the failures that cost a runner round trip: non-ASCII text, a parameter list that
differs from meta.json, a register used past its declared count, a branch to a missing label.
"""

import json
import subprocess
import sys
import unittest

from tests.helpers import ROOT, lint_ptx

FOLDERS = ("problems", "problems_extra", "problems_asm")


def problem_kernels():
    for folder in FOLDERS:
        for meta_path in sorted((ROOT / folder).glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            for kernel in meta["kernels"]:
                yield meta_path.parent, kernel


class GeneratedPtx(unittest.TestCase):
    def test_every_kernel_compiles_to_wellformed_ptx(self):
        checked = 0
        for problem, kernel in problem_kernels():
            with self.subTest(problem=problem.name, kernel=kernel["entry"]):
                ir = next(problem.glob(f"{kernel['entry']}.ir.json"))
                result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-400:])
                self.assertEqual(lint_ptx(result.stdout, kernel["entry"], kernel["arg_order"]), [])
                checked += 1
        self.assertGreater(checked, 0)

    def test_stdout_is_ptx_only_even_with_dump_graph(self):
        problem = ROOT / "problems" / "p00_scalar_add"
        ir = next(problem.glob("*.ir.json"))
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir), "--dump-graph"],
                                capture_output=True, text=True)
        self.assertTrue(result.stdout.startswith(".version"))
        self.assertIn("kernel", result.stderr)

    def test_lint_catches_what_it_claims_to(self):
        good = ".version 8.0\n.visible .entry k(\n        .param .u64 param_x\n)\n{\n        .reg .b32 %r<2>;\nDONE:\n        ret;\n}\n"
        self.assertEqual(lint_ptx(good, "k", ["x"]), [])
        self.assertTrue(lint_ptx(good.replace("%r<2>", "%r<1>").replace("ret;", "mov.u32 %r1, 0;\n ret;"), "k", ["x"]))
        self.assertTrue(lint_ptx(good.replace("ret;", "bra NOWHERE;"), "k", ["x"]))
        self.assertTrue(lint_ptx(good, "k", ["y"]))
        self.assertTrue(lint_ptx(good + "é", "k", ["x"]))


class Examples(unittest.TestCase):
    def test_committed_examples_are_current(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "make_examples.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "run tools/make_examples.py\n" + result.stdout)


if __name__ == "__main__":
    unittest.main()
