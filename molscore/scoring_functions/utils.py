from typing import Union
import subprocess
import threading
import os
import signal
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors, rdmolops, DataStructs
from rdkit.Avalon import pyAvalonTools
from xarray import DataArray


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


class Fingerprints:
    """
    Class to organise Fingerprint generation
    """

    @classmethod
    def get_fp(cls, mol: Union[str, Chem.rdchem.Mol], name: str, nBits: int, asarray: bool = False):
        """
        Get fp by str instead of method
        :param mol: RDKit mol or Smiles
        :param name: Name of FP [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, hashAP, hashTT, RDK5, RDK6, RDK7]
        :param nBits: Number of bits
        :return:
        """
        mol = get_mol(mol)
        fp = None
        for m in [cls.ECFP4, cls.ECFP4c, cls.FCFP4, cls.FCFP4c,
                  cls.ECFP6, cls.ECFP6c, cls.FCFP6, cls.FCFP6c,
                  cls.Avalon, cls.MACCSkeys, cls.hashAP, cls.hashTT,
                  cls.RDK5, cls.RDK6, cls.RDK7]:
            if name == m.__name__: fp = m

        if (mol is not None) and (fp is not None):
            fp = m(mol, nBits, asarray)

        if fp is None:
            raise KeyError(f"{name} name not recognised as a valid fingerprint")

        return fp

    # Circular fingerprints
    @staticmethod
    def ECFP4(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)

    @staticmethod
    def ECFP4c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True)

    @staticmethod
    def FCFP4(mol, nBits, asarray):
        if asarray:
            np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True)

    @staticmethod
    def FCFP4c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True, useFeatures=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True, useFeatures=True)

    @staticmethod
    def ECFP6(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits)

    @staticmethod
    def ECFP6c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)

    @staticmethod
    def FCFP6(mol, nBits, asarray):
        if asarray:
            np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True)

    @staticmethod
    def FCFP6c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)

    # Substructure fingerprints
    @staticmethod
    def Avalon(mol, nBits, asarray):
        if asarray:
            return np.asarray(pyAvalonTools.GetAvalonFP(mol, nBits=nBits))
        else:
            return pyAvalonTools.GetAvalonFP(mol, nBits=nBits)

    @staticmethod
    def MACCSkeys(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMACCSKeysFingerprint(mol))
        else:
            return rdMolDescriptors.GetMACCSKeysFingerprint(mol)

    # Path-based fingerprints
    @staticmethod
    def hashAP(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits))
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)

    @staticmethod
    def hashTT(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits))
        else:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)

    @staticmethod
    def RDK5(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK6(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK7(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)


def GetSimilarityMeasure(name: str, bulk=False):
    """
    Helper function to get correct RDKit similarity function by name
    :param name: RDKit similarity type
    :param bulk: Whether get bulk similarity
    """
    similarity_function = None

    if bulk:
        for f in [
            DataStructs.BulkAllBitSimilarity, DataStructs.BulkAsymmetricSimilarity, DataStructs.BulkBraunBlanquetSimilarity,
            DataStructs.BulkCosineSimilarity, DataStructs.BulkMcConnaugheySimilarity, DataStructs.BulkDiceSimilarity,
            DataStructs.BulkKulczynskiSimilarity, DataStructs.BulkRusselSimilarity, DataStructs.BulkOnBitSimilarity,
            DataStructs.BulkRogotGoldbergSimilarity, DataStructs.BulkSokalSimilarity, DataStructs.BulkTanimotoSimilarity
            ]:
            if "Bulk" + name + "Similarity" == f.__name__:
                similarity_function = f
    else:
        for f in [
            DataStructs.AllBitSimilarity, DataStructs.AsymmetricSimilarity, DataStructs.BraunBlanquetSimilarity,
            DataStructs.CosineSimilarity, DataStructs.McConnaugheySimilarity, DataStructs.DiceSimilarity,
            DataStructs.KulczynskiSimilarity, DataStructs.RusselSimilarity, DataStructs.OnBitSimilarity,
            DataStructs.RogotGoldbergSimilarity, DataStructs.SokalSimilarity, DataStructs.TanimotoSimilarity
            ]:
            if name + "Similarity" == f.__name__:
                similarity_function = f

    if similarity_function is None:
        raise KeyError(f"{name} not found")

    return similarity_function