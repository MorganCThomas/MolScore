import warnings
from functools import partial

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from moleval.metrics.metrics_utils import (
    SillyWalks,
    get_mol,
    mapper,
    mol_passes_filters,
)
from moleval.utils import Fingerprints


class ChemistryFilter:
    def __init__(self, target: list = None, n_jobs=1):
        self.n_jobs = n_jobs
        # Preprocess target
        self.target = target
        self.silly_walks = None
        if self.target:
            # SillyWalks
            self.ref_sillyness = SillyWalks(self.target, n_jobs=n_jobs)
            # MolWt, LogP
            self.ref_MW = self._compute_mean_std(self._mols2prop(self.target, "MolWt"))
            self.ref_LogP = self._compute_mean_std(
                self._mols2prop(self.target, "MolLogP")
            )
            self.target_property_filters = [
                partial(
                    self.property_filter,
                    prop="MolWt",
                    min=(self.ref_MW[0] - 4 * self.ref_MW[1]),
                    max=(self.ref_MW[0] + 4 * self.ref_MW[1]),
                ),
                partial(
                    self.property_filter,
                    prop="MolLogP",
                    min=(self.ref_LogP[0] - 4 * self.ref_LogP[1]),
                    max=(self.ref_LogP[0] + 4 * self.ref_LogP[1]),
                ),
                partial(
                    self.sillyness_filter,
                    ref_sillyness=self.ref_sillyness,
                    threshold=0.1,
                ),
            ]

    @staticmethod
    def MolWt(x):
        mol = get_mol(x)
        if mol:
            return Descriptors.MolWt(mol)

    @staticmethod
    def MolLogP(x):
        mol = get_mol(x)
        if mol:
            return Descriptors.MolLogP(mol)

    @staticmethod
    def property_filter(mol, prop, min, max):
        mol = get_mol(mol)
        if mol:
            value = getattr(Descriptors, prop)(mol)
            if (value >= min) and (value <= max):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def sillyness_filter(mol, ref_sillyness, threshold):
        return ref_sillyness.score(mol)[0] <= threshold

    def _mols2prop(self, mols, prop):
        tfunc = getattr(self, prop)
        r = [x for x in mapper(self.n_jobs)(tfunc, mols) if x is not None]
        return r

    @staticmethod
    def _compute_mean_std(x):
        return np.mean(x), np.std(x)

    @staticmethod
    def passes_basic(mol):
        passes = mol_passes_filters(
            mol=mol,
            allowed=None,  # Means allowed atoms are {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
            allow_charge=True,
            isomericSmiles=False,
            molwt_min=150,
            molwt_max=650,
            mollogp_max=4.5,
            rotatable_bonds_max=7,
            filters=True,  # MOSES MCF and PAINS filters
        )
        return passes

    @staticmethod
    def passes_target(mol, property_filters):
        mol = get_mol(mol)
        if (mol is not None) and all([f(mol) for f in property_filters]):
            return True
        else:
            return False

    def filter_molecule(self, mol, basic=True, target=False):
        mol = get_mol(mol)
        passes = False
        if mol:
            passes_basic = True
            passes_target = True
            if basic:
                passes_basic = self.passes_basic(mol)
            if target:
                passes_target = self.passes_target(
                    mol, property_filters=self.target_property_filters
                )
            passes = passes_basic and passes_target
        return passes

    def filter_molecules(self, mols, basic=True, target=False):
        func = partial(self.filter_molecule, basic=basic, target=target)
        results = mapper(self.n_jobs)(func, mols)
        return results


class ChemistryBuffer:
    def __init__(self, buffer_size=100, diversity_threshold=0.35):
        self.buffer_size = buffer_size
        self.diversity_threshold = 0.35
        self.buffer = pd.DataFrame([], columns=["centroid", "score", "fp", "members"])
        self.filtered = set()

    def add_molecule(self, idx: int, mol: str, score: float):
        # Keep memory of already process molecules
        if idx in self.filtered:
            self.filtered.add(idx)
            return

        # Fill the buffer first
        if len(self.buffer) < self.buffer_size:
            fp = Fingerprints.get_fp(name="ECFP4", mol=mol, nBits=2048)
            if fp:
                row = pd.Series(
                    [
                        idx,
                        score,
                        fp,
                        [],
                    ],
                    index=self.buffer.columns,
                )
                self.buffer.loc[len(self.buffer)] = row
                self.filtered.add(idx)
            return

        # If score isn't good ignore too
        if score < self.buffer.score.min():
            self.filtered.add(idx)
            return

        # Compute fingerprint and similarity
        fp = Fingerprints.get_fp(name="ECFP4", mol=mol, nBits=2048)
        if fp is None:
            return
        sims = BulkTanimotoSimilarity(fp, list(self.buffer.fp))
        cidx = np.argmax(sims)

        # Either add as member or update centroid and score
        if sims[cidx] > self.diversity_threshold:
            if score > self.buffer.iloc[cidx]["score"]:
                # NOTE choice to not update centroid and fp here, which could lead to impure clusters
                # Update the best score of cluster
                self.buffer.iat[cidx, 1] = score
            # Add as member
            self.buffer.iat[cidx, 3].append(idx)
        else:
            # Add as new centroid replacing lowest scoring which should be purged
            row = pd.Series([idx, score, fp, []], index=self.buffer.columns)
            self.buffer.loc[self.buffer.score.idxmin()] = row

        self.filtered.add(idx)

    def update_from_score_metrics(self, df, endpoint):
        for idx, row in df.iterrows():
            self.add_molecule(idx, row["smiles"], row[endpoint])

    def top_n(self, n):
        """Return the top n score for molecules in the buffer"""
        if n > self.buffer_size:
            warnings.warn(f"n of {n} is greater than buffer of size {len(self.buffer)}")
        return (
            self.buffer.sort_values("score", ascending=False).iloc[:n]["score"].mean()
        )
