import os
from multiprocessing import Pool
from functools import partial
import pickle as pkl
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import gzip
from Levenshtein import distance as levenshtein

from rdkit import Chem, SimDivFilters, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Cluster import Butina
from rdkit import rdBase
from molvs.standardize import Standardizer


from moleval.metrics.metrics_utils import mol_passes_filters

rdBase.DisableLog('rdApp.*')


def butina_cs(fps, distThresh, reordering=False):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    matrix = []
    for i in tqdm(range(1, nfps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
        matrix.append(sims)

    # now cluster the data:
    cs = Butina.ClusterData(data=dists, nPts=nfps, distThresh=distThresh, isDistData=True, reordering=reordering)
    return cs


def se_cs(fps, distThresh):
    lp = SimDivFilters.rdSimDivPickers.LeaderPicker()
    picks = lp.LazyBitVectorPick(fps, len(fps), distThresh)

    cs = defaultdict(list)
    # Assign each centroid as first item in list
    for i, idx in enumerate(picks):
        cs[i].append(idx)
    # Prepare similarity matrix
    sims = np.zeros((len(picks), len(fps)))
    # For each pick
    for i in range(len(picks)):
        pick = picks[i]
        # Assign bulk similarity to row
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
        # Assign similarity to self as 0, so as not to pick yourself
        sims[i, i] = 0
    # Find snn to each pick
    best = np.argmax(sims, axis=0)
    # For each snn
    for i, idx in enumerate(best):
        # If it's not already a centroid
        if i not in picks:
            # Assign to nearest centroid...
            cs[idx].append(i)
    return [cs[k] for k in cs]


def leven_butina_cs(smiles, distThresh=3, reordering=False):
    cs = Butina.ClusterData(data=smiles, nPts=len(smiles), distThresh=distThresh,
                            distFunc=levenshtein, reordering=reordering)
    return cs


def butina_picker(dataset: list, input_format='smiles', n=3,
                  threshold=0.65, radius=2, nBits=1024, selection='largest', return_cs=False):
    """
    Select a subset of molecules and return a list of (RDKit mol centroid, size of cluster, optional(clusters))
    tuples.

    :param dataset: List of SMILES or rdkit mols
    :param input_format: Whether the dataset is of 'smiles' or 'mol' type
    :param n: Number of molecules to pick
    :param threshold: Tanimoto Distance threshold for clusters assignment
    :param radius: Morgan fingerprint radius
    :param nBits: Morgan fingerprint bit length
    :param selection: Whether to return centroids from the 'largest' clusters, 'smallest' clusters or a 'range'
    of clusters size (Evenly spread between max and min sizes depending on n)
    :param return_cs: Return a full list of clusters (mols)
    :return (centroids, clusters sizes, optional(list of clusters))
    """

    if input_format == 'smiles':
        mols = [Chem.MolFromSmiles(smi) for smi in dataset if Chem.MolFromSmiles(smi)]
    elif input_format == 'mol':
        mols = dataset
    else:
        print('Format not recognized')
        raise

    assert selection in ['largest', 'smallest', 'range']

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in mols]

    cs = butina_cs(fps, threshold)

    # Return subset
    if selection == 'largest':
        cs = sorted(cs, key=lambda x: len(x), reverse=True)
        ids = []
        size = []
        for i in range(n):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if selection == 'smallest':
        cs = sorted(cs, key=lambda x: len(x), reverse=False)
        ids = []
        size = []
        for i in range(n):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if selection == 'range':
        cs = sorted(cs, key=lambda x: len(x), reverse=False)
        ids = []
        size = []
        for i in np.linspace(0, len(cs) - 1, n).astype(np.int64):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if return_cs:
        cs_subset = [[mols[i] for i in c] for c in cs]
        return subset, size, cs_subset
    else:
        return subset, size


def se_picker(dataset: list, input_format='smiles', n=3,
              threshold=0.65, radius=2, nBits=1024, selection='largest', n_jobs=1, return_cs=False):
    if input_format == 'smiles':
        mols = [Chem.MolFromSmiles(smi) for smi in dataset if Chem.MolFromSmiles(smi)]
    elif input_format == 'mol':
        mols = dataset
    else:
        print('Format not recognized')
        raise

    assert selection in ['largest', 'smallest', 'range']

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in mols]

    cs = se_cs(fps, threshold)

    # Return subset
    if selection == 'largest':
        cs = sorted(cs, key=lambda x: len(x), reverse=True)
        ids = []
        size = []
        for i in range(n):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if selection == 'smallest':
        cs = sorted(cs, key=lambda x: len(x), reverse=False)
        ids = []
        size = []
        for i in range(n):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if selection == 'range':
        cs = sorted(cs, key=lambda x: len(x), reverse=False)
        ids = []
        size = []
        for i in np.linspace(0, len(cs) - 1, n).astype(np.int64):
            ids.append(cs[i][0])
            size.append(len(cs[i]))
        subset = [mols[i] for i in ids]

    if return_cs:
        cs_subset = [[mols[i] for i in c] for c in cs]
        return subset, size, ids, cs_subset
    else:
        return subset, size, ids


def maxmin_picker(dataset: list, input_format='smiles', n=3, seed=123, radius=2, nBits=1024):
    """
    Select a subset of molecules and return a list of diverse RDKit mols.
    http://rdkit.blogspot.com/2014/08/optimizing-diversity-picking-in-rdkit.html
    """

    if input_format == 'smiles':
        mols = [Chem.MolFromSmiles(smi) for smi in dataset if Chem.MolFromSmiles(smi)]
    elif input_format == 'mol':
        mols = dataset
    else:
        print('Format not recognized')
        raise

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in mols]

    mmp = SimDivFilters.MaxMinPicker()
    ids = mmp.LazyBitVectorPick(fps, len(fps), n)
    subset = [mols[i] for i in ids]

    return subset


def single_nearest_neighbour(fp, fps):
    """
    Return the max Tanimoto coefficient and index of single nearest neighbour
    """
    Tc_vec = DataStructs.cDataStructs.BulkTanimotoSimilarity(fp, fps)
    Tc = np.max(Tc_vec)
    idx = np.argmax(Tc_vec)
    return Tc, idx


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode('utf-8') for smi in smiles]
    else:
        with open(file_path, 'rt') as f:
            smiles = f.read().splitlines()
    return smiles


def write_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (os.path.dirname(file_path) != ''):
        os.makedirs(os.path.dirname(file_path))
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        with gzip.open(file_path, 'wb') as f:
            _ = [f.write((smi+'\n').encode('utf-8')) for smi in smiles]
    else:
        with open(file_path, 'wt') as f:
            _ = [f.write(smi+'\n') for smi in smiles]
    return


def read_pickle(file):
    with open(file, 'rb') as f:
        x = pkl.load(f)
    return x


def write_pickle(object, file):
    d = os.path.dirname(file)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(file, 'wb') as f:
        pkl.dump(object, f)
    return


def canonize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        smi = Chem.MolToSmiles(mol)
        return smi
    else:
        return


def canonize_list(smiles, n_jobs=1):
    with Pool(n_jobs) as pool:
        can_smiles = [smi for smi in tqdm(pool.imap(canonize, smiles), total=len(smiles)) if smi is not None]

    return can_smiles


def neutralize_atoms(smi, isomericSmiles=False):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            try:
                for at_idx in at_matches_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    chg = atom.GetFormalCharge()
                    hcount = atom.GetTotalNumHs()
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(hcount - chg)
                    atom.UpdatePropertyCache()
                smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
                return smiles
            except:
                return None
        else:
            return Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    else:
        return None


def process_smi(smi, isomeric, moses_filters, neutralize, **filter_kwargs):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol:
        stand_mol = Standardizer().fragment_parent(mol)
        can_smi = Chem.MolToSmiles(stand_mol, isomericSmiles=isomeric)
        if moses_filters:
            if not mol_passes_filters(can_smi, isomericSmiles=isomeric, allow_charge=True, **filter_kwargs):
                return None
        # Modification to original code
        if neutralize:
            can_smi = neutralize_atoms(can_smi, isomericSmiles=isomeric)

    else:
        return None
    return can_smi


def process_list(smiles, isomeric, moses_filters, neutralize, n_jobs=1, **filter_kwargs):
    with Pool(n_jobs) as pool:
        pprocess = partial(process_smi, isomeric=isomeric, moses_filters=moses_filters,
                           neutralize=neutralize, **filter_kwargs)
        proc_smiles = [smi for smi in tqdm(pool.imap(pprocess, smiles), total=len(smiles)) if smi is not None]
    return proc_smiles


class Fingerprints:

    @classmethod
    def check_mol(cls, mol):
        if isinstance(mol, str):
            return Chem.MolFromSmiles(mol)
        if isinstance(mol, Chem.rdchem.Mol):
            return mol
        else:
            print("Unknown mol format")
            raise

    # Circular fingerprints
    @classmethod
    def ECFP4(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)

    @classmethod
    def ECFP4_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits))

    @classmethod
    def ECFP4c(cls, mol):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprint(mol, 2, useCounts=True)

    @classmethod
    def ECFP4c_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2, useCounts=True)
        nfp = np.zeros((1, nBits), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % nBits
            nfp[0, nidx] += int(v)
        return nfp.reshape(-1)

    @classmethod
    def FCFP4(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True)

    @classmethod
    def FCFP4_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True))

    @classmethod
    def FCFP4c(cls, mol):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)

    @classmethod
    def FCFP4c_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, nBits), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % nBits
            nfp[0, nidx] += int(v)
        return nfp.reshape(-1)

    @classmethod
    def ECFP6(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits)

    @classmethod
    def ECFP6_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits))

    @classmethod
    def ECFP6c(cls, mol):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)

    @classmethod
    def ECFP6c_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 3, useCounts=True)
        nfp = np.zeros((1, nBits), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % nBits
            nfp[0, nidx] += int(v)
        return nfp.reshape(-1)

    @classmethod
    def FCFP6(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True)

    @classmethod
    def FCFP6_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True))

    @classmethod
    def FCFP6c(cls, mol):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)

    @classmethod
    def FCFP6c_arr(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, nBits), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % nBits
            nfp[0, nidx] += int(v)
        return nfp.reshape(-1)

    # Structural fingerprints
    @classmethod
    def Avalon(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return pyAvalonTools.GetAvalonFP(mol, nBits=nBits)

    @classmethod
    def MACCSkeys(cls, mol):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)

    # Path-based fingerprints
    @classmethod
    def hashAP(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)

    @classmethod
    def hashTT(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)

    @classmethod
    def RDK5(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)

    @classmethod
    def RDK6(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)

    @classmethod
    def RDK7(cls, mol, nBits):
        mol = Fingerprints.check_mol(mol)
        return rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)

