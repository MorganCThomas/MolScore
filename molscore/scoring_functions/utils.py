from typing import Union
import subprocess
import threading
import os
import signal
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem import AllChem as Chem


class timedThread(object):
    """
    Subprocess wrapped into a thread to add a more well defined timeout, use os to send a signal to all PID in
    group just to be sure... (nothing else was doing the trick)
    """

    def __init__(self, timeout):
        self.cmd = None
        self.timeout = timeout
        self.process = None

    def run(self, cmd):
        self.cmd = cmd.split()
        def target():
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self.process.communicate()
            return

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self.timeout)
        if thread.is_alive():
            print('Process timed out...')
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        return


class timedSubprocess(object):
    """
    Currently used
    """
    def __init__(self, timeout, shell=False):
        self.cmd = None
        self.timeout = timeout
        assert isinstance(self.timeout, float)
        self.shell = shell
        self.process = None

    def run(self, cmd):
        if not self.shell:
            self.cmd = cmd.split()
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.cmd = cmd
            self.process = subprocess.Popen(self.cmd, shell=self.shell,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print('Process timed out...')
            if not self.shell:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.kill()
        return


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    return


def get_mol(mol: Union[str, Chem.rdchem.Mol]):
    """
    Get RDkit mol
    :param mol:
    :return: RDKit Mol
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        raise TypeError

    if not mol:
        mol = None

    return mol
