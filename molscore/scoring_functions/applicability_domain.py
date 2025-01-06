import logging
import os
from functools import partial
from typing import Union

import numpy as np
from rdkit.Chem import QED, Crippen, Descriptors, FindMolChiralCenters
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3
from rdkit.Chem.Scaffolds import MurckoScaffold

from molscore.scoring_functions.utils import Fingerprints, Pool, get_mol

logger = logging.getLogger("applicability_domain")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class ApplicabilityDomain:
    """
    Measures if a molecule is within the applicability domain of a reference dataset as described in
    Maxime Langevin et al. in "Impact of applicability domains to generative artificial intelligence"
    10.26434/chemrxiv-2022-mdhwz
    Note: Use several of me to combine different similarity-feature combinations
    Note: Only range is computed here, MolecularSimilarity can be used to compute distance to reference FPs
    """

    return_metrics = ["in_AD"]

    def __init__(
        self,
        prefix,
        ref_smiles: Union[os.PathLike, str, list],
        fp: str = None,
        qed: bool = False,
        physchem: bool = True,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: A file_path to or list of reference smiles used to define the applicability domain.
        :param QED: Whether to include QED
        :param fp: Type of fingerprint used to featurize the molecule [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, AP, hashAP, hashTT, RDK5, RDK6, RDK7, PHCO]
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        """
        self.prefix = prefix.replace(" ", "_")
        self.fp = fp
        self.qed = qed
        self.physchem = physchem
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)

        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            with open(ref_smiles, "r") as f:
                self.ref_smiles = f.read().splitlines()
        else:
            assert isinstance(ref_smiles, list) and (
                len(ref_smiles) > 0
            ), "None list or empty list provided"
            self.ref_smiles = ref_smiles

        # Calculate features
        logger.info("Computing reference features")
        mols = [m for m in self.mapper(get_mol, self.ref_smiles) if m is not None]
        pfunc = partial(
            self.compute_features, fp=self.fp, qed=self.qed, physchem=self.physchem
        )
        self.ref_features = np.asarray([f for f in self.mapper(pfunc, mols)])

        # Compound bounds
        self.ref_max = np.max(self.ref_features, axis=0)
        self.ref_min = np.min(self.ref_features, axis=0)

    @staticmethod
    def compute_physchem(mol: Union[Chem.rdchem.Mol, str]) -> list:
        """
        Compute the physchem descriptors as described in the original implementation
        """
        mol = get_mol(mol)
        descriptors = [
            QED.qed,
            Descriptors.NumHDonors,
            Descriptors.NumHAcceptors,
            Descriptors.RingCount,
            Descriptors.NumRotatableBonds,
            Descriptors.TPSA,
            Crippen.MolLogP,
            Crippen.MolMR,
            Descriptors.MolWt,
            CalcFractionCSP3,
            Descriptors.HeavyAtomCount,
        ]
        physchem_features = [d(mol) for d in descriptors]
        # Add BM/HA ratio custom features
        bm_mol = MurckoScaffold.GetScaffoldForMol(mol)
        physchem_features.append(
            Descriptors.HeavyAtomCount(bm_mol) / Descriptors.HeavyAtomCount(mol)
        )
        # Add ring info
        ri = mol.GetRingInfo().AtomRings()
        n_ring, max_ring, min_ring = (
            len(ri),
            len(max(ri, key=len, default=())),
            len(min(ri, key=len, default=())),
        )
        physchem_features += [n_ring, max_ring, min_ring]
        # Add charge info
        charges = []
        for a in mol.GetAtoms():
            charges.append(a.GetFormalCharge())
        physchem_features += [sum(charges), max(charges), min(charges)]
        # Add chiral centres
        physchem_features.append(len(FindMolChiralCenters(mol, includeUnassigned=True)))
        return physchem_features

    @staticmethod
    def compute_features(
        mol: Union[Chem.rdchem.Mol, str],
        fp: str = None,
        qed: bool = False,
        physchem: bool = True,
    ) -> list:
        """
        Compute features as specified during initialisation
        """
        mol = get_mol(mol)
        if mol:
            mol_features = None

            if fp:
                fp = list(Fingerprints.get(mol, name=fp, nBits=1024, asarray=True))
                if mol_features is None:
                    mol_features = fp
                else:
                    mol_features += fp

            if qed:
                qed = QED.qed(mol)
                if mol_features is None:
                    mol_features = [qed]
                else:
                    mol_features += [qed]

            if physchem:
                physchems = ApplicabilityDomain.compute_physchem(mol)
                if mol_features is None:
                    mol_features = physchems
                else:
                    mol_features += physchems
            return mol_features

    @staticmethod
    def score_smi(smi, prefix, ref_max, ref_min, fp, qed, physchem):
        """
        Calculate the score for a single smiles
        """
        result = {"smiles": smi}
        # Compute features
        smi_features = ApplicabilityDomain.compute_features(
            smi, fp=fp, qed=qed, physchem=physchem
        )
        if smi_features is not None:
            # Check domain
            above_min = ref_min <= smi_features
            below_max = ref_max >= smi_features
            in_domain = above_min & below_max
            result.update({f"{prefix}_in_AD": float(in_domain.all())})
        else:
            result.update(
                {f"{prefix}_{m}": 0.0 for m in ApplicabilityDomain.return_metrics}
            )
        return result

    def score(self, smiles: list, **kwargs) -> list:
        """
        Calculate the binary scores representing whether smiles are within the AD (1.0) or not (0.0)
        :param smiles: List of SMILES strings
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Compute features
        pfunc = partial(
            self.score_smi,
            prefix=self.prefix,
            ref_max=self.ref_max,
            ref_min=self.ref_min,
            fp=self.fp,
            qed=self.qed,
            physchem=self.physchem,
        )
        results = [r for r in self.mapper(pfunc, smiles)]
        return results

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate the binary scores representing whether smiles are within the AD (1.0) or not (0.0)
        :param smiles: List of SMILES strings
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles)
