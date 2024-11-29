import gzip
import multiprocessing
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
import atexit
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from dask.distributed import Client, LocalCluster
from func_timeout import FunctionTimedOut, func_timeout
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, rdMolDescriptors, rdmolops
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


# ----- Requirements related -----
def check_openbabel():
    if shutil.which("obabel") is None:
        raise RuntimeError(
            "OpenBabel is required for this function, please install it using conda or mamba e.g., mamba install openbabel -c conda-forge"
        )


def check_exe(command):
    if shutil.which(command) is None:
        return False
    else:
        return True


def check_env(key):
    if key in os.environ.keys():
        return True
    else:
        return False


def check_path(path):
    if os.path.exists(path):
        return True
    else:
        return False


# ----- Multiprocessing related -----
def Pool(n_jobs, return_map=True, **kwargs):
    if platform.system() == "Linux":
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context("spawn")

    # Extract from environment as default, overriding configs
    if "MOLSCORE_NJOBS" in os.environ.keys():
        pool = context.Pool(int(os.environ["MOLSCORE_NJOBS"]))
        atexit.register(pool.terminate)
        return pool.imap if return_map else pool
    
    # Transform float into fraction of CPUs
    if isinstance(n_jobs, float):
        cpu_count = os.cpu_count()
        n_jobs = int(cpu_count * n_jobs)
    
    # Return pool or None
    if isinstance(n_jobs, int) and (n_jobs > 1):
        pool = context.Pool(n_jobs, *kwargs)
        atexit.register(pool.terminate)
        return pool.imap if return_map else pool
    else:
        return map if return_map else pool


def test_func():
    mol = Chem.MolFromSmiles(
        "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
    )
    Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs=100)
    Chem.MMFFOptimizeMoleculeConfs(mol)
    return True


class timedFunc:
    @staticmethod
    def _func_wrapper(func, child_conn):
        try:
            result = func()
            child_conn.send(result)
        except Exception as e:
            child_conn.send(e)
        child_conn.close()

    def __init__(self, func, timeout: Union[int, float]):
        """
        Wrap a function by a timeout clause to also ideally work with C++ bindings i.e, RDKit
        Based on: https://stackoverflow.com/questions/51547126/timeout-a-c-function-from-python
        :param func: A function to be run with timeout
        :param timeout: Timeout
        """
        self.func = func
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        """
        Run the timeout wrapped function with input args and kwargs
        :param args: Function args
        :param kwargs: Function kwargs
        :return: Function result or None if timedout
        """
        pfunc = partial(self.func, *args, **kwargs)
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(
            target=self._func_wrapper, args=(pfunc, child_conn)
        )
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return None  # Return None if the timeout was reached

        result = parent_conn.recv()
        parent_conn.close()
        if isinstance(result, Exception):
            raise result
        return result


class timedFunc2:
    def __init__(self, func, timeout: Union[int, float]):
        """
        Wrap a function by a timeout using timed_func which should also work with C++ bindings i.e, RDKit
        :param func: A function to be run with timeout
        :param timeout: Timeout
        """
        self.func = func
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        """
        Run the timeout wrapped function with input args and kwargs
        :param args: Function args
        :param kwargs: Function kwargs
        :return: Function result or None if timedout
        """
        try:
            result = func_timeout(self.timeout, self.func, args, kwargs)
        except FunctionTimedOut:
            return None

        return result


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
            self.process = subprocess.Popen(
                self.cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = self.process.communicate()
            return

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self.timeout)
        if thread.is_alive():
            print("Process timed out...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        return


class timedSubprocess(object):
    """
    Currently used
    """

    def __init__(self, timeout=None, shell=False):
        self.cmd = None
        self.cwd = None
        self.timeout = timeout
        self.shell = shell
        self.process = None

    def run(self, cmd, cwd=None):
        if not self.shell:
            self.cmd = cmd.split()
            self.process = subprocess.Popen(
                self.cmd,
                preexec_fn=os.setsid,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self.cmd = cmd
            self.process = subprocess.Popen(
                self.cmd,
                shell=self.shell,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        try:
            out, err = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print("Process timed out...")
            out, err = (
                "".encode(),
                f"Timed out at {self.timeout}".encode(),
            )  # Encode for consistency
            if not self.shell:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()
        return out, err


class DaskUtils:
    # TODO add dask-jobqueue templates https://jobqueue.dask.org/en/latest/
    # TODO add dask-ssh cluster

    @classmethod
    def setup_dask(
        cls, cluster_address_or_n_workers=None, local_directory=None, logger=None
    ):
        client = None
        client = cls._setup_from_environment(local_directory=local_directory)
        if client is None:
            client = cls._setup_from_arguments(
                cluster_address_or_n_workers, local_directory=local_directory
            )
        if (
            (logger is not None)
            and (client is None)
            and (cluster_address_or_n_workers is not None)
        ):
            logger.warning(f"Unrecognized dask input: {cluster_address_or_n_workers}")
        return client

    @staticmethod
    def _local_client(n_workers: float, local_directory=None):
        cluster = LocalCluster(
            n_workers=int(n_workers),
            processes=True,
            threads_per_worker=1,
            local_directory=local_directory,
        )
        client = Client(cluster)
        print(f"Dask worker dashboard: {client.dashboard_link}")
        # ---- Export to env for further scoring functions to also connect to -----
        os.environ["MOLSCORE_CLUSTER"] = client.scheduler.address
        # --------------------------------------------------------------------------
        return client

    @staticmethod
    def _distributed_client(address: str):
        client = Client(address)
        print(f"Dask worker dashboard: {client.dashboard_link}")
        return client

    @staticmethod
    def _slurm_client(cores, memory="1GB", queue=None, local_directory=None):
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError:
            raise ImportError(
                "pip install dask-jobqueue is required if you want to use slurm clusters"
            )

        # Optional environment parameters
        if "MOLSCORE_SLURM_QUEUE" in os.environ.keys():
            queue = os.environ["MOLSCORE_SLURM_QUEUE"]
        if "MOLSCORE_SLURM_MEMORY" in os.environ.keys():
            memory = os.environ["MOLSCORE_SLURM_MEMORY"]

        # Setup cluster
        cluster = SLURMCluster(
            queue=queue,
            n_workers=cores,
            cores=1,
            memory=memory,
            local_directory=local_directory,
        )
        client = Client(cluster)
        print(f"Dask worker dashboard: {client.dashboard_link}")
        # ---- Export to env for further scoring functions to also connect to -----
        os.environ["MOLSCORE_CLUSTER"] = client.scheduler.address
        # --------------------------------------------------------------------------
        return client

    @staticmethod
    def _environment_slurm():
        if ("MOLSCORE_SLURM_CORES" in os.environ.keys()) or (
            "MOLSCORE_SLURM_CPUS" in os.environ.keys()
        ):
            return os.environ["MOLSCORE_SLURM_CPUS"]
        else:
            return False

    @staticmethod
    def _environment_address():
        if "MOLSCORE_CLUSTER" in os.environ.keys():
            return os.environ["MOLSCORE_CLUSTER"]
        else:
            return False

    @staticmethod
    def _environment_njobs():
        if "MOLSCORE_NJOBS" in os.environ.keys():
            return int(os.environ["MOLSCORE_NJOBS"])
        else:
            return False

    @classmethod
    def _setup_from_environment(cls, local_directory=None):
        env_cluster = cls._environment_address()
        env_njobs = cls._environment_njobs()
        env_slurm = cls._environment_slurm()
        if env_cluster:
            print(
                f"Identified an environment cluster address ({env_cluster}), this overrides any config parameters."
            )
            client = cls._distributed_client(env_cluster)
            nworkers = len(client.scheduler_info()["workers"])
            print(
                f"Connected to scheduler {env_cluster} with {nworkers} workers"
            )  # , to change this behaviour remove this variable via <unset MOLSCORE_CLUSTER>")
        elif env_slurm:
            print(
                f"Identified an environment specifying {env_slurm} SLURM cores, this overrides any config parameters."
            )
            client = cls._slurm_client(
                cores=int(env_slurm), local_directory=local_directory
            )
            time.sleep(5)  # Ugly wait to spin up cluster
            nworkers = len(client.scheduler_info()["workers"])
            print(f"SLURM cluster created with {nworkers} workers")
        elif env_njobs:
            print(
                f"Identified an environment specifying {env_njobs} workers, this overrides any config parameters."
            )
            client = cls._local_client(
                n_workers=int(env_njobs), local_directory=local_directory
            )
            nworkers = len(client.scheduler_info()["workers"])
            print(
                f"Local cluster created with {nworkers} workers"
            )  # , to change this behaviour remove this variable via <unset MOLSCORE_NJOBS>")
        else:
            client = None
        return client

    @classmethod
    def _setup_from_arguments(
        cls, cluster_address_or_n_workers=None, local_directory=None
    ):
        if isinstance(cluster_address_or_n_workers, str):
            client = cls._distributed_client(cluster_address_or_n_workers)
        elif isinstance(cluster_address_or_n_workers, float) or isinstance(
            cluster_address_or_n_workers, int
        ):
            client = cls._local_client(cluster_address_or_n_workers)
        else:
            client = None
        return client

    @staticmethod
    def _close_dask(client):
        if client:
            client.close()
            # If local cluster close that too, can't close remote cluster
            try:
                client.cluster.close()
            except Exception:
                pass

# ----- Chemistry related -----
def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")
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
        raise TypeError("Molecule is not a string (SMILES) or rdkit.mol")

    if not mol:
        mol = None

    return mol


def canonize_smiles(smiles_or_mol, return_none=True):
    mol = get_mol(smiles_or_mol)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        if return_none:
            return None
        else:
            return smiles_or_mol


def read_mol(mol_path: os.PathLike, i=0):
    if mol_path.endswith(".mol2") or mol_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(mol_path, sanitize=False, strictParsing=False)

    elif mol_path.endswith(".sdf"):
        suppl = Chem.ForwardSDMolSupplier(mol_path, sanitize=False)
        for i, mol in enumerate(suppl):
            if i == i:
                break

    elif mol_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(mol_path, sanitize=False)

    else:
        raise TypeError(f"Cannot read molecule, unknown input file type: {mol_path}")

    return mol


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode("utf-8") for smi in smiles]
    else:
        with open(file_path, "rt") as f:
            smiles = f.read().splitlines()
    return smiles


def write_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (
        os.path.dirname(file_path) != ""
    ):
        os.makedirs(os.path.dirname(file_path))
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        with gzip.open(file_path, "wb") as f:
            _ = [f.write((smi + "\n").encode("utf-8")) for smi in smiles]
    else:
        with open(file_path, "wt") as f:
            _ = [f.write(smi + "\n") for smi in smiles]
    return


class Fingerprints:
    """
    Class to organise Fingerprint generation
    """

    @staticmethod
    def get(
        mol: Union[str, Chem.rdchem.Mol], name: str, nBits: int, asarray: bool = False
    ):
        """
        Get fp by str instead of method
        :param mol: RDKit mol or Smiles
        :param name: Name of FP [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, AP, hashAP, hashTT, RDK5, RDK6, RDK7, PHCO]
        :param nBits: Number of bits
        :return:
        """
        mol = get_mol(mol)
        generator = getattr(Fingerprints, name, None)

        if generator is None:
            raise KeyError(f"'{name}' not recognised as a valid fingerprint")

        if mol is not None:
            return generator(mol, nBits, asarray)

    # Circular fingerprints
    @staticmethod
    def ECFP4(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=nBits
                )
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=nBits
            )

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
            np.asarray(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=nBits, useFeatures=True
                )
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=nBits, useFeatures=True
            )

    @staticmethod
    def FCFP4c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(
                mol, radius=2, useCounts=True, useFeatures=True
            )
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(
                mol, radius=2, useCounts=True, useFeatures=True
            )

    @staticmethod
    def ECFP6(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=3, nBits=nBits
                )
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=3, nBits=nBits
            )

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
            np.asarray(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=3, nBits=nBits, useFeatures=True
                )
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=3, nBits=nBits, useFeatures=True
            )

    @staticmethod
    def FCFP6c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(
                mol, radius=3, useCounts=True, useFeatures=True
            )
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(
                mol, radius=3, useCounts=True, useFeatures=True
            )

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
    def AP(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10)

    @staticmethod
    def hashAP(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
            )
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=nBits
            )

    @staticmethod
    def hashTT(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                    mol, nBits=nBits
                )
            )
        else:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=nBits
            )

    @staticmethod
    def RDK5(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)
            )
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK6(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)
            )
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK7(mol, nBits, asarray):
        if asarray:
            return np.asarray(
                rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)
            )
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)

    # Pharmacophore-based
    @staticmethod
    def PHCO(mol, nBits, asarray):
        if asarray:
            return np.asarray(Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory))
        else:
            return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


class SimilarityMeasures:
    @staticmethod
    def get(name, bulk=False):
        """
        Helper function to get correct RDKit similarity function by name
        :param name: RDKit similarity type [AllBit, Asymmetric, BraunBlanquet, Cosine, McConnaughey, Dice, Kulczynski, Russel, OnBit, RogotGoldberg, Sokal, Tanimoto]
        :param bulk: Whether get bulk similarity
        """
        if bulk:
            name = "Bulk" + name + "Similarity"
        else:
            name = name + "Similarity"

        similarity_function = getattr(DataStructs, name, None)
        if similarity_function is None:
            raise KeyError(f"'{name}' not found")

        return similarity_function
