"""End to end on the hosted runner: compile, assemble with ptxas, launch on the GPU, compare outputs.

Opt in with EMMY_RUNNER_TESTS=1 (needs network). Without it these tests are skipped, so the rest of the
suite stays offline.
"""

import os
import subprocess
import sys
import unittest

from tests.helpers import ROOT

ENABLED = os.environ.get("EMMY_RUNNER_TESTS") == "1"


@unittest.skipUnless(ENABLED, "set EMMY_RUNNER_TESTS=1 to run against the hosted runner")
class OnTheRunner(unittest.TestCase):
    def run_all(self, *args):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "run_all.py"), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])

    def test_open_problem_set(self):
        self.run_all("problems")

    def test_probe_problems(self):
        self.run_all("problems_extra")

    def test_tma_kernel_assembles(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "check_assembles.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout[-800:])


if __name__ == "__main__":
    unittest.main()
