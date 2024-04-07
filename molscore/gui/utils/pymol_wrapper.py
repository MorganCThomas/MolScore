import os
import subprocess


class PyMol:
    def __init__(self, pymol_path: str, pymol_args=["-pqQ"]):
        """
        :type pymol_path: str
        :param pymol_args: Args
        """
        self._pymol_path = os.path.join(pymol_path, "pymol")
        self._pymol_args = pymol_args
        self._pymol_command = [pymol_path] + pymol_args
        self._initPipe()

    def _initPipe(self):
        cmd = [self._pymol_path] + self._pymol_args
        self._pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE).stdin

    def do(self, cmmd):
        if not isinstance(cmmd, bytes):
            cmmd = cmmd.encode("utf-8")
        try:
            self._pipe.write(cmmd + b"\n")
            self._pipe.flush()
        except OSError:
            return False
        return True

    def __call__(self, cmmd):
        return self.do(cmmd)

    def close(self):
        self.do("quit")
        self._pipe.close()
