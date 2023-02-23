# coding=utf-8
"""
Adapted from
https://github.com/tblaschke/reinvent-memory
"""

import abc
import json
import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Scaffolds import MurckoScaffold

from molscore.scaffold_memory.ScaffoldMemory import ScaffoldMemory


class ScaffoldFilter(ScaffoldMemory):

    def __init__(self, nbmax=25, minscore=0.6, generic=False, outputmode="binary"):
        super(ScaffoldFilter, self).__init__()
        self.nbmax = nbmax  # number of smiles for one scaffold to score until the penalizer starts
        self.minscore = minscore  # only add smiles with a minimum score into the memory
        self.generic = generic  # store generic scaffolds or normal murcko scaffolds?
        self._scaffoldfunc = self.getGenericScaffold if generic else self.getScaffold
        self._outputmode = outputmode

    @abc.abstractmethod
    def score(self, smiles, scores_dict: dict) -> np.array:
        raise NotImplemented

    def validScores(self, smiles, scores) -> bool:
        if not len(smiles) == len(scores):
            logging.error("SMILES and score vector are not the same length. Do nothing")
            logging.debug(smiles)
            logging.debug(scores)
            return False
        else:
            return True

    def savetojson(self, file):
        savedict = {'nbmax':      self.nbmax, 'minscore': self.minscore, 'generic': self.generic,
                    "_scaffolds": self._scaffolds}
        jsonstr = json.dumps(savedict, sort_keys=True, indent=4, separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def savetocsv(self, file):
        df = {"Cluster": [], "Scaffold": [], "SMILES": []}
        for i, scaffold in enumerate(self._scaffolds):
            for smi, score in self._scaffolds[scaffold].items():
                df["Cluster"].append(i)
                df["Scaffold"].append(scaffold)
                df["SMILES"].append(smi)
                for item in score.keys():
                    if item in df:
                        df[item].append(score[item])
                    else:
                        df[item] = [score[item]]

        df = pd.DataFrame(df)
        df.to_csv(file, index=False)

    def _sigmoid(self, x, k=0.15):
        # sigmoid function
        # use k to adjust the slope
        x = x*2 -1 
        s = 1 / (1 + np.exp(-x / k)) 
        return s

    
    def calculate_output(self, nb_in_bucket: int):
        if nb_in_bucket == 0:
            return 1
        if nb_in_bucket > self.nbmax:
            return 0    
        if nb_in_bucket <= self.nbmax:
            frac = nb_in_bucket/self.nbmax
            if self._outputmode == "sigmoid":
                return 1 - self._sigmoid(frac)
            elif self._outputmode == "linear":
                return 1 - frac
            else:  #self._outputmode == "binary"
                return 1

            
class ScaffoldMatcher(ScaffoldFilter):
    def __init__(self, nbmax=25, minscore=0.6, generic=False, outputmode="binary"):
        super().__init__(nbmax=nbmax, minscore=minscore, generic=generic, outputmode=outputmode)

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            try:
                scaffold = self._scaffoldfunc(smile)
            except Exception:
                scaffold = ''
                scores[i] = 0
            if self.has(scaffold, smile):
                scores[i] = 0
            elif score >= self.minscore:
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                self._update_memory([smile], [scaffold], [save_score])
                scores[i] = scores[i] * self.calculate_output(len(self[scaffold]))
        return scores

    def savetojson(self, file):
        savedict = {'nbmax':      self.nbmax, 'minscore': self.minscore, 'generic': self.generic,
                    "_scaffolds": self._scaffolds}
        jsonstr = json.dumps(savedict, sort_keys=True, indent=4, separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)


class IdenticalMurckoScaffold(ScaffoldMatcher):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated. 'minsimilarity' is ignored."""

    def __init__(self, nbmax=25, minscore=0.6, outputmode="binary", **kwargs):
        """
        :param nbmax: Maximum number of molecules per memory bin (cluster)
        :param minscore: Minimum molecule score required to consider for memory binning
        :param outputmode: 'binary' (1 or 0), 'linear' (1 - fraction of bin) or 'sigmoid' (1 - sigmoid(fraction of bin)) [binary, linear, sigmoid]
        :param kwargs:
        """
        super().__init__(nbmax=nbmax, minscore=minscore, generic=False, outputmode=outputmode)


class IdenticalTopologicalScaffold(ScaffoldMatcher):
    """Penalizes compounds based on exact Topological Scaffolds previously generated. 'minsimilarity' is ignored."""

    def __init__(self, nbmax=25, minscore=0.6, outputmode="binary", **kwargs):
        """
        :param nbmax: Maximum number of molecules per memory bin (cluster)
        :param minscore: Minimum molecule score required to consider for memory binning
        :param outputmode: 'binary' (1 or 0), 'linear' (1 - fraction of bin) or 'sigmoid' (1 - sigmoid(fraction of bin)) [binary, linear, sigmoid]
        :param kwargs:
        """
        super().__init__(nbmax=nbmax, minscore=minscore, generic=True, outputmode=outputmode)


class CompoundSimilarity(ScaffoldFilter):
    """Penalizes compounds based on the ECFP or FCFP Tanimoto similarity to previously generated compounds."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6, radius=2, useFeatures=False,
                 bits=2048, outputmode="binary", **kwargs):
        """
        :param nbmax: Maximum number of molecules per memory bin (cluster)
        :param minscore: Minimum molecule score required to consider for memory binning
        :param minsimilarity: Minimum similarity to centroid molecule in bin
        :param radius: Morgan fingerprint radius (e.g., 2 = ECFP4)
        :param useFeatures: Include feature information in fingerprint
        :param bits: Length of fingerprint (i.e., number of folded bits)
        :param outputmode: 'binary' (1 or 0), 'linear' (1 - fraction of bin) or 'sigmoid' (1 - sigmoid(fraction of bin)) [binary, linear, sigmoid]
        :param kwargs:
        """
        super().__init__(nbmax=nbmax, minscore=minscore, generic=False, outputmode=outputmode)
        self.minsimilarity = minsimilarity
        self.radius = radius
        self.useFeatures = useFeatures
        self.bits = bits

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            if score >= self.minscore:
                cluster, fingerprint, isnewcluster = self.findCluster(smile)
                if self.has(cluster, smile):
                    scores[i] = 0
                    continue
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                if isnewcluster:
                    self._update_memory([smile], [cluster], [save_score], [fingerprint])
                else:
                    self._update_memory([smile], [cluster], [save_score])
                scores[i] = scores[i] * self.calculate_output(len(self[cluster]))

        return scores

    def findCluster(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "", "", False
        if self.bits > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.bits, useFeatures=self.useFeatures)
        else:
            fp = AllChem.GetMorganFingerprint(mol, self.radius, useFeatures=self.useFeatures)

        if smiles in self.getFingerprints():
            return smiles, fp, False

        fps = list(self.getFingerprints().values())
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if len(sims) == 0:
            return smiles, fp, True
        closest = np.argmax(sims)
        if sims[closest] >= self.minsimilarity:
            return list(self.getFingerprints().keys())[closest], fp, False
        else:
            return smiles, fp, True


class ScaffoldSimilarityAtomPair(CompoundSimilarity):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6, outputmode="binary", **kwargs):
        """
        :param nbmax: Maximum number of molecules per memory bin (cluster)
        :param minscore: Minimum molecule score required to consider for memory binning
        :param minsimilarity: Minimum similarity to centroid molecule in bin
        :param outputmode: 'binary' (1 or 0), 'linear' (1 - fraction of bin) or 'sigmoid' (1 - sigmoid(fraction of bin)) [binary, linear, sigmoid]
        :param kwargs:
        """
        super().__init__(nbmax=nbmax, minscore=minscore, minsimilarity=minsimilarity, outputmode=outputmode)

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            if score >= self.minscore:
                cluster, fingerprint, isnewcluster = self.findCluster(smile)
                if self.has(cluster, smile):
                    scores[i] = 0
                    continue
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                if isnewcluster:
                    self._update_memory([smile], [cluster], [save_score], [fingerprint])
                else:
                    self._update_memory([smile], [cluster], [save_score])
                scores[i] = scores[i] * self.calculate_output(len(self[cluster]))

        return scores

    def findCluster(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            except:
                return "", "", False
            if scaffold:
                cluster = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            else:
                return "", "", False
        else:
            return "", "", False

        fp = Pairs.GetAtomPairFingerprint(scaffold)  # Change to Tanimoto?
        if cluster in self.getFingerprints():
            return cluster, fp, False

        fps = list(self.getFingerprints().values())
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if len(sims) == 0:
            return cluster, fp, True
        closest = np.argmax(sims)
        if sims[closest] >= self.minsimilarity:
            return list(self.getFingerprints().keys())[closest], fp, False
        else:
            return cluster, fp, True


class ScaffoldSimilarityECFP(CompoundSimilarity):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, nbmax: int = 50, minscore: float = 0.5, minsimilarity: float = 0.8, radius: int = 2,
                 useFeatures: bool = False, bits: int = 1024, outputmode: str = "linear", **kwargs):
        """
        :param nbmax: Maximum number of molecules per memory bin (cluster)
        :param minscore: Minimum molecule score required to consider for memory binning
        :param minsimilarity: Minimum similarity to centroid molecule in bin
        :param radius: Morgan fingerprint radius (e.g., 2 = ECFP4)
        :param useFeatures: Include feature information in fingerprint
        :param bits: Length of fingerprint (i.e., number of folded bits)
        :param outputmode: 'binary' (1 or 0), 'linear' (1 - fraction of bin) or 'sigmoid' (1 - sigmoid(fraction of bin)) [binary, linear, sigmoid]
        :param kwargs:
        """
        super().__init__(nbmax=nbmax, minscore=minscore, minsimilarity=minsimilarity, outputmode=outputmode)
        self.radius = radius
        self.useFeatures = useFeatures
        self.bits = bits

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            if score >= self.minscore:
                cluster, fingerprint, isnewcluster = self.findCluster(smile)
                if self.has(cluster, smile):
                    scores[i] = 0
                    continue
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                if isnewcluster:
                    self._update_memory([smile], [cluster], [save_score], [fingerprint])
                else:
                    self._update_memory([smile], [cluster], [save_score])
                scores[i] = scores[i] * self.calculate_output(len(self[cluster]))

        return scores

    def findCluster(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            except:
                return "", "", False
            if scaffold:
                cluster = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            else:
                return "", "", False
        else:
            return "", "", False

        if self.bits > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(scaffold, self.radius, nBits=self.bits,
                                                       useFeatures=self.useFeatures)
        else:
            fp = AllChem.GetMorganFingerprint(scaffold, self.radius, useFeatures=self.useFeatures)

        if smiles in self.getFingerprints():
            return smiles, fp, False

        fps = list(self.getFingerprints().values())
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if len(sims) == 0:
            return cluster, fp, True
        closest = np.argmax(sims)
        if sims[closest] >= self.minsimilarity:
            return list(self.getFingerprints().keys())[closest], fp, False
        else:
            return cluster, fp, True


class NoScaffoldFilter(ScaffoldFilter):
    """Don't penalize compounds. Only save them with more than 'minscore'. All other arguments are ignored."""
    def __init__(self, minscore=0.6, minsimilarity=0.6, nbmax=25, outputmode="binary"):
        super().__init__(minscore=minscore)

    def score(self, smiles, scores_dict: dict) -> np.array:
        """
        we only log the compounds
        """
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            try:
                scaffold = self._scaffoldfunc(smile)
            except Exception:
                scaffold = ''
            if score >= self.minscore:
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                self._update_memory([smile], [scaffold], [save_score])
        return scores
