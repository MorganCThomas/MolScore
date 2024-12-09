import multiprocessing
import os
import platform
from collections import Counter, defaultdict
from functools import partial
from itertools import combinations
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from rdkit import Chem, SimDivFilters
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold

from moleval.metrics.ifg import identify_functional_groups
from moleval.metrics.NP_Score import npscorer
from moleval.metrics.quality_filters import rd_filters
from moleval.metrics.SA_Score import sascorer

_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, "mcf.csv"))
_pains = pd.read_csv(
    os.path.join(_base_dir, "wehi_pains.csv"), names=["smarts", "names"]
)
_alerts_path = os.path.join(_base_dir, "quality_filters", "alert_collection.csv")
_guacamol_rules_path = os.path.join(_base_dir, "quality_filters", "guacamol_rules.json")
_filters = [
    Chem.MolFromSmarts(x)
    for x in pd.concat([_mcf, _pains], axis=0, sort=True)["smarts"].values
]


def mapper(n_jobs):
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool False, will return its map function
    """
    if n_jobs == 1:

        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, float):
        n_jobs = int(multiprocessing.cpu_count() * n_jobs)
        assert (
            0 < n_jobs <= multiprocessing.cpu_count()
        ), f"n_jobs must be between 0 and {multiprocessing.cpu_count()}"
    if isinstance(n_jobs, int):
        if platform.system() == "Linux":
            context = multiprocessing.get_context("fork")
        else:
            context = multiprocessing.get_context("spawn")

        if "MOLEVAL_NJOBS" in os.environ.keys():
            pool = context.Pool(int(os.environ["MOLEVAL_NJOBS"]))
        else:
            pool = context.Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def get_mol(smiles_or_mol):
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    else:
        return None


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    can_smiles = Chem.MolToSmiles(mol)
    if Chem.MolFromSmiles(
        can_smiles
    ):  # Check it can be converted back, rarely something goes wrong
        return can_smiles
    else:
        return None


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    mol = get_mol(mol)  #
    if mol is None:  #
        return None  #
    fgs = AllChem.FragmentOnBRICSBonds(mol)  # get_mol(mol) -> mol
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def functional_groups(mol):
    mol = get_mol(mol)
    if mol is None:  #
        return None  #
    else:
        fgs = identify_functional_groups(
            mol
        )  # each fg is a named tuple (atomIdx, atoms, atoms+environment)
    return [fg[2] for fg in fgs]


def compute_functional_groups(mol_list, n_jobs=1):
    fgs = Counter()
    for mol_groups in mapper(n_jobs)(functional_groups, mol_list):
        if mol_groups is not None:
            fgs.update(mol_groups)
    return fgs


def ring_systems(mol):
    mol = get_mol(mol)
    if mol is None:  #
        return None  #
    else:
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon:  # Making conscious choice to include spiro
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems

        # Get ring system canonical smiles
        system_smiles = []
        for ring in systems:
            mw = Chem.RWMol()
            atom_map = {}
            # Build atoms
            for idx in ring:
                atom_map.update({idx: mw.GetNumAtoms()})  # Will be index of added atom
                mw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum()))
            # Build bonds
            for i, j in combinations(ring, 2):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    mw.AddBond(atom_map[i], atom_map[j], bond.GetBondType())
            system_smiles.append(Chem.MolToSmiles(mw))

    return system_smiles


def compute_ring_systems(mol_list, n_jobs=1):
    rss = Counter()
    for mol_systems in mapper(n_jobs)(ring_systems, mol_list):
        if mol_systems is not None:
            rss.update(mol_systems)
    return rss


def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    if None in fragments:  #
        fragments.pop(None)  #
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scaffold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    if mol is None:  #
        return None  #
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == "" or n_rings < min_rings:
        return None
    else:  #
        return scaffold_smiles  #


def average_agg_tanimoto(
    stock_vecs, gen_vecs, batch_size=5000, agg="max", device="cpu", p=1
):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (
                (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
                .cpu()
                .numpy()
            )
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0)
                )
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    return agg_tanimoto


def analogues_tanimoto(
    stock_vecs, gen_vecs, batch_size=5000, similarity_threshold=0.4, device="cpu"
):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        similarity_threshold: Molecules above this threshold are considered analogues
    """
    stock_analogues = np.zeros(len(stock_vecs))
    gen_analogues = np.zeros(len(gen_vecs))

    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (
                (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
                .cpu()
                .numpy()
            )
            jac[np.isnan(jac)] = 1

            # Compute number below threshold
            jac_thresh = np.where(jac > similarity_threshold, 1, 0)
            gen_analogues[i : i + y_gen.shape[1]] = np.maximum(
                gen_analogues[i : i + y_gen.shape[1]], jac_thresh.max(0)
            )  # max will be 1 if any analogue
            stock_analogues[j : j + x_stock.shape[0]] = np.maximum(
                stock_analogues[j : j + x_stock.shape[0]], jac_thresh.max(1)
            )

    return gen_analogues, stock_analogues


def fingerprint(
    smiles_or_mol,
    fp_type="maccs",
    dtype=None,
    morgan__r=2,
    morgan__n=1024,
    *args,
    **kwargs,
):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == "maccs":
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype="uint8")
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == "morgan":
        fingerprint = np.asarray(
            Morgan(molecule, morgan__r, nBits=morgan__n), dtype="uint8"
        )
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    """
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    """
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(partial(fingerprint, *args, **kwargs), smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [
        fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
        for fp in fps
    ]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def numpy_fp_to_bitvector(fp):
    """
    Doesn't currently handle np.NaN
    :param fp:
    :return:
    """
    if (np.isnan(fp)).any():
        return None
    bit_vector = Chem.DataStructs.ExplicitBitVect(len(fp))
    for i, v in enumerate(fp):
        if v:
            bit_vector.SetBit(i)
    return bit_vector


def numpy_fps_to_bitvectors(fps, n_jobs=1):
    bit_vectors = mapper(n_jobs)(partial(numpy_fp_to_bitvector), fps)
    return list(set(bit_vectors) - {None})


def sphere_exclusion(fps, dist_thresh=0.65):
    """
    Roger Sayle's algorithm for picking diverse compounds, analogous to sphere exclusion.
    :param fps: Must be of ExplicitBitVect type
    :param dist_thresh:
    :return:
    """
    lp = SimDivFilters.rdSimDivPickers.LeaderPicker()
    ids = lp.LazyBitVectorPick(fps, len(fps), dist_thresh)
    return len(ids)


def mol_passes_filters(
    mol,
    allowed=None,
    allow_charge=True,
    isomericSmiles=False,
    molwt_min=150,
    molwt_max=650,
    mollogp_max=4.5,
    rotatable_bonds_max=7,
    filters=True,
):
    """
    MOSES defaults
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    * is between 250 - 350 Da
    * is not greater than 7 rotatable bonds
    * XlogP < 3.5
    """
    allowed = allowed or {"C", "N", "S", "O", "F", "Cl", "Br", "H"}
    mol = get_mol(mol)
    if mol is None:
        return False
    # Large rings
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(len(x) >= 8 for x in ring_info.AtomRings()):
        return False
    # Charge
    h_mol = Chem.AddHs(mol)
    if not allow_charge:
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            return False
    # Atoms
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    # MolWt
    if not molwt_min <= Descriptors.MolWt(mol) <= molwt_max:
        return False
    # LogP
    if not Descriptors.MolLogP(mol) <= mollogp_max:
        return False
    # Rotatable Bonds
    if not Descriptors.NumRotatableBonds(mol) <= rotatable_bonds_max:
        return False
    # MCF, PAINS filters
    if filters:
        if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
            return False

    # RDKit parse check
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True


class QualityFilter:
    def __init__(self, rules: str = _guacamol_rules_path, n_jobs=1):
        # Load files
        self.rf = rd_filters.RDFilters(_alerts_path)
        rule_dict = rd_filters.read_rules(rules)
        rule_list = [
            x.replace("Rule_", "")
            for x in rule_dict.keys()
            if x.startswith("Rule") and rule_dict[x]
        ]
        rule_str = " and ".join(rule_list)
        self.rf.build_rule_list(rule_list)
        self._n_jobs = n_jobs

    def score(self, mol):
        output = self.rf.evaluate(mol)
        if output[0] == "OK":
            return True
        else:
            return False

    @staticmethod
    def _score(mol, rf):
        output = rf.evaluate(mol)
        if output[0] == "OK":
            return True
        else:
            return False

    def score_mols(self, mols, normalize=True, n_jobs=1):
        mols = [m for m in mapper(self._n_jobs)(get_mol, mols) if m is not None]
        scores = [
            s for s in mapper(self._n_jobs)(partial(self._score, rf=self.rf), mols)
        ]
        if normalize:
            return np.mean(scores)
        else:
            return np.sum(scores)


def neutralize_atoms(mol, isomericSmiles=False):
    mol = get_mol(mol)
    if mol:
        pattern = Chem.MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
        )
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


def MCF_PAINS_filters(mol):
    """
    Mol passes MCF and Pains filters
    :param mol: Smiles or RDKit.Mol
    :return: bool
    """
    mol = get_mol(mol)
    if mol is None:
        return False
    h_mol = Chem.AddHs(mol)

    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    else:
        return True


class SillyWalks:
    """From https://github.com/PatWalters/silly_walks"""

    def __init__(self, reference_mols, n_jobs=1):
        self.count_dict = defaultdict(int)
        self._n_jobs = n_jobs
        bit_counts = mapper(self._n_jobs)(self.count_bits, reference_mols)
        for count_dict in bit_counts:
            for k, v in count_dict.items():
                self.count_dict[k] += v

    @staticmethod
    def count_bits(mol):
        count_dict = {}
        mol = get_mol(mol)
        if mol is not None:
            fp = AllChem.GetMorganFingerprint(mol, 2)
            for k, v in fp.GetNonzeroElements().items():
                count_dict[k] = v
        return count_dict

    def score(self, mol: Union[str, Chem.rdchem.Mol]) -> Tuple:
        """
        Return the ratio of outlier FP bits in a molecule
        :param mol: SMILES or RDKit Mol
        :return: (Ratio of outlier FP bits, identifies bits, molecule bit information)
        """
        mol = get_mol(mol)
        if mol is not None:
            bi = {}
            fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=bi)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [bit for bit in on_bits if self.count_dict[bit] == 0]
            score = len(silly_bits) / len(on_bits)
            return score, silly_bits, bi

    @staticmethod
    def _score(mol: Union[str, Chem.rdchem.Mol], count_dict) -> Tuple:
        mol = get_mol(mol)
        if mol is not None:
            bi = {}
            fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=bi)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [bit for bit in on_bits if count_dict[bit] == 0]
            score = len(silly_bits) / len(on_bits)
            return score, silly_bits, bi

    def score_mols(self, mols: list) -> float:
        """
        Return the average ratio of outlier FP bits for a list of molecules
        :param: List of SMILES or RDKit Mols
        :return: Scores for all molecules
        """
        mols = [m for m in mapper(self._n_jobs)(get_mol, mols) if m is not None]
        scores = [
            s[0]
            for s in mapper(self._n_jobs)(
                partial(self._score, count_dict=self.count_dict), mols
            )
        ]
        return scores
