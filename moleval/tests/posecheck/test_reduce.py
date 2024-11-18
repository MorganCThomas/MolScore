import os
import subprocess
import unittest

from moleval.metrics.posecheck.utils.constants import EXAMPLE_PDB_PATH, REDUCE_PATH


class TestReduce(unittest.TestCase):
    def test_reduce_exists(self):
        command = REDUCE_PATH
        exit_response = subprocess.run(
            command, capture_output=True, text=True, shell=True
        )
        self.assertEqual(exit_response.returncode, 1)

    def test_reduce_executable(self):
        command = REDUCE_PATH + f" -NOFLIP {EXAMPLE_PDB_PATH}"
        exit_response = subprocess.run(
            command, capture_output=True, text=True, shell=True
        )
        self.assertEqual(exit_response.returncode, 0)
