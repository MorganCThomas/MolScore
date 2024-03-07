# coding=utf-8
"""
Adapted from
https://github.com/tblaschke/reinvent-memory
"""

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

rdBase.DisableLog('rdApp.error')


class ScaffoldMemory(object):
    def __init__(self):
        self._scaffolds = {} ## Dictionary of scaffold: {smiles: score}
        self._morganfp = {} ## Dictionary of scaffold: morganfp

    def add(self, smiles, scores=None):
        if scores:
            assert len(smiles) == len(scores), "Score vector is not the same length as SMILES list"
        scaffolds = [self.getScaffold(smi) for smi in smiles]
        self._update_memory(smiles, scaffolds, scores)
        return scaffolds

    def addGeneric(self, smiles, scores=None):
        if scores:
            assert len(smiles) == len(scores), "Score vector is not the same length as SMILES list"
        scaffolds = [self.getGenericScaffold(smi) for smi in smiles]
        self._update_memory(smiles, scaffolds, scores)
        return scaffolds

    def getScaffold(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        else:
            return ''

    def getGenericScaffold(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        else:
            return ''

    def _update_memory(self, smiles, scaffolds, scores=None, fingerprints=None):
        for i, smi in enumerate(smiles):
            scaffold = scaffolds[i]
            if fingerprints is not None:
                self._morganfp[scaffold] = fingerprints[i]
            score = scores[i]
            if scaffold in self._scaffolds:
                self._scaffolds[scaffold][smi] = score
            else:
                self._scaffolds[scaffold] = {smi: score}

    def has(self, scaffold, smiles):
        if scaffold in self._scaffolds:
            if smiles in self._scaffolds[scaffold]:
                return True
        return False

    def getFingerprints(self):
        return self._morganfp

    def __getitem__(self, scaffold):
        if scaffold in self._scaffolds:
            return self._scaffolds[scaffold]
        else:
            return []
