import logging
import re

import numpy as np
from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions.utils import Pool, get_mol

logger = logging.getLogger("reaction_filter")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class DecoratedReactionFilter:
    """Score molecules based on whether scaffold decorations adhere to a set of reaction filters"""

    return_metrics = ["score"]

    LibINVENT_reactions = [
        "[#6;$(C[C;$(C([#6]))]):4]-!@[N;$([NH1;D2](C)C);!$(N-[#6]=[*]);$(N([C])):3]>>[#6:4][*].[N:3][*]",
        "[C;$([CH;$(C([#6])[#6])]),$([CH2;$(C[#6])]):1]-!@[N;$(N(C=O)C=O):2]>>[*:1][*].[*:2][*]",
        "[C;$([CH;$(C([#6])[#6])]),$([CH2;$(C[#6])]):1]-!@[O;$(Oc1ccccc1):2]>>[*:1][*].[*:2][*]",
        "[C;$([CH;$(C([#6])[#6])]),$([CH2;$(C[#6])]):1]-!@[N;$(N([#6])S(=O)=O):2]>>[*:1][*].[*:2][*]",
        "[S;$(S(=O)(=O)[C,N]):1]-!@[N+0;$(NC):2]>>[*:1][*].[*:2][*]",
        "[N;$(N-[#6]):3]-!@[C;$(C=O):1]-!@[N+0;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND1]);!$(N[O,N]):2]>>[*:1][*].[*:2][*]",
        "[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][c:2]>>[*:2][*].[*:1][*]",
        "[#6;$(C=[#6!H0]):1][C;$(C#N):2]>>[*:1][*].[*][*:2]",
        "[#6:1]([N+]([O-])=O)=[#6:2]>>[*:1][*][N+]([O-])=O.[*:2][*]",
        "[#6;!$(A(A=[O,S]));!$(A=*);!$([A-]);!$(A~[P,S,O,N]):3][C:1](=[#7:2])[N!H0;!$(A(A=[O,S]));!$(A=*);!$([A-]);!$(A~[P,S,O,N]):4]>>[#6:3][C:1]([*])=[N:2].[#7!H0:4][*]",
        "[#6;!$(C(C=*)(C=*));!$([#6]~[O,N,S]);$([#6]~[#6]):1][C:2](=[O:3])[N;D2;$(N(C=[O,S]));!$(N~[O,P,S,N]):4][#6;!$(C=*);!$([#6](~[O,N,S])N);$([#6]~[#6]):5]>>[#6:1][C:2](=[O:3])[*].[*][N:4][#6:5]",
        "[#6;!R;!$(C=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][#6;!R;!$(C=*);!$([#6]~[O,N,S]);$([#6]~[#6]):2]>>[#6:1][*].[#6:2][*]",
        "[N;!H0:1]([C:2]([#7:5][#6:6])=[#8:3])[#6:4]>>[#8:3]=[C:2]([#7:1][#6:4])[*].[*][#7:5][#6:6]",
        "[#6;!$(C(C=*)(C=*));!$([#6]~[O,N,S]);$([#6]~[#6]):1][C:2](=[O:3])[N;D2;$(N(C=[O,S]));!$(N~[O,P,S,N]):4][#6;!$(C=*);!$([#6](~[O,N,S])N);$([#6]~[#6]):5]>>[#6:1][C:2](=[O:3])[*].[*][N:4][#6:5]",
        "[#6;!$([#6]=*);!$([#6]~[O,N,S,P]);$([#6]~[#6]):2]-!@[#6;!$([#6]=*);!$([#6]~[O,N,S,P]);$([#6]~[#6]):1]>>[#6;$([#6]~[#6]);!$([#6]~[S,N,O,P]):1][*].[*][#6;$([#6]~[#6]);!$([#6]~[S,N,O,P]):2]",
        "[CH2;$([#6]~[#6]):4]-!@[O:3]-!@[#6;$([#6]~[#6]);!$([#6]=O):2]>>[#6;$([#6]~[#6]);!$([#6]=O):2][#8][*].[*][#6;H2;$([#6]~[#6]):4]",
        "[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]",
        "[*;$(c2aaaaa2),$(c2aaaa2):4]/[#6:1]=!@[#6:2]/[*;$(c2aaaaa2),$(c2aaaa2):3]>>[#6;c,$(C(=O)O),$(C#N):3][#6;H1:2]=[#6;H1:1][*].[#6;$([#6]=[#6]),$(c:c):4][*]",
        "[#6:4][#6;H0:1]=!@[#6:2]([#6:5])[#6:3]>>[#6;c,$(C(=O)O),$(C#N):3][#6:2]([#6:5])=[#6;$([#6][#6]):1][*].[#6;$([#6]=[#6]),$(c:c):4][*]",
        "[*;$(c);$(C=C-[#6]),$(c):1]-!@[*;$(c):2]>>[#6;$(C=C-[#6]),$(c):1][*].[*][*;$(c):2]",
        "[C;$(C([#6])[#6]):1]([#6:5])([#6:2])([O;H1:3])[#6;!R:4]>>[#6:2][#6:1](*)([#6:5])[O:3].[*][#6:4]",
        "[#6;$(C=C-[#6]),$(c:c):1]-!@[C;$(C#CC):2]>>[#6;$(C=C-[#6]),$(c:c):1][*].[*][CH1;$(C#CC):2]",
        "[c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1]-!@[N;$(NC)&!$(N=*)&!$([N-])&!$(N#*)&!$([ND1])&!$(N[O])&!$(N[C,S]=[S,O,N]),H2&$(Nc1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):2]>>[*][c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1].[*][N:2]",
        "[*;!$(c1ccccc1);$(c1[n,c]c[n,c]c[n,c]1):1]-!@[N;$(NC);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(n[c,O]);!$(N[C,S]=[S,O,N]):2]>>[*;!$(c1ccccc1);$(c1[n,c]c[n,c]c[n,c]1):1][*].[*][N:2]",
        "[*;$(c1c(N(~O)~O)cccc1):1]-!@[N;$(NC);!$(N=*);!$([N-]);!$(N#*);!$([ND1]);!$(N[O]);!$(N[C,S]=[S,O,N]):2]>>[*;$(c1c(N(~O)~O)cccc1):1][*].[*][N:2]",
        "[*;$(c1ccc(N(~O)~O)cc1):1]-!@[N;$(NC);!$(N=*);!$([N-]);!$(N#*);!$([ND1]);!$(N[O]);!$(N[C,S]=[S,O,N]):2]>>[*;$(c1ccc(N(~O)~O)cc1):1][*].[*][N:2]",
        "[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):2]>>[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):1][*].[#6;!$([#6]=*);!$([#6]~[O,N,S]);$([#6]~[#6]):2][*]",
        "[C:2]([#7;!D4:1])(=[O:3])[#6:4]>>[#7:1][*].[C,$(C=O):2](=[O:3])([*])[#6:4]",
        "[#6;$(C(=O)):1][#7,#8,#16:2]>>[*:1][*].[*:2][*]",
        "[O:2]=[#6:1][#7:5]>>[O:2]=[#6:1][*].[N:5][*]",
        "[#6;$(C=[O]):1][#8,#16:2]>>[*:1][*].[*][*:2]",
        "[N;!$(n1****1);!$(n1*****1);!$(N=*);!$(N(A=A));!$([N-]);!$(N~[O,P,S,N]):1]-!@[#6;!$(C=*);!$(C(A=A));!$([C-]);!$(C~[O,P,S]):2]>>[N:1][*].[*][#6:2]",
        "[#6:8][O:7][C:5](=[O:6])[C:4]([C:2](=[O:3])[#6:1])[#6:9]>>[#6:1][C:2]([C:4]([*])[C:5]([O:7][#6:8])=[O:6])=[O:3].[#6:9][*]",
        "[#6:1][C:2]([#6:7])[C:3](=[O:4])[O:5][#6:6]>>[C;!H0:2]([*])([C:3]([O:5][#6:6])=[O:4])[#6:1].[#6:7][*]",
        "[N;!$(n1****1);!$(n1*****1);!$(N(A=A));!$(N=*);!$([N-]);!$(N~[O,P,S,N]):1][*;$(c1aaaaa1),$(c1aaaa1);!$(C=*);!$(C(A=A));!$([C-]);!$(C~[O,P,S]):2]>>[N:1][*].[#6:2][*]",
        "[C:3]([C:1]([#8:5][#6:6])=[O:2])[#6:7]=[O:8]>>[#6:6][#8:5][C:1](=[O:2])[C!H0:3][*].[#6:7](=[O:8])[*]",
        "[N+:1]([#6:2])([#6:4])([#6:5])[#6:3]>>[N;!$(N=*);!$([N-]);!$(N~[O,P,S,N]):1]([#6:2])([#6:3])([*])[#6:4].[*][#6:5]",
        "[c:1][C,N,S,O:2]>>[c:1][*].[*:2]",
    ]

    def __init__(
        self,
        prefix: str,
        scaffold: str,
        custom_reactions: list = None,
        libinvent_reactions: bool = True,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Name given to scoring function
        :param scaffold: Assumes de novo molecule generation is decorative from this scaffold without attachment points e.g., (benzene=c1ccccc1)
        :param custom_reactions: Provide a custom list of SMIRKS
        :param libinvent_reactions: Run pre-defined reaction filters from LibINVENT
        :param n_jobs: Number of parallel jobs to run
        """
        self.prefix = prefix.replace(" ", "_")
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.reaction_smirks = []
        self.scaffold = get_mol(scaffold)
        assert self.scaffold, f"Error parsing scaffold {scaffold}"
        if custom_reactions:
            self.reaction_smirks.extend(custom_reactions)
        if libinvent_reactions:
            self.reaction_smirks.extend(self.LibINVENT_reactions)

        self.reactions = [Chem.ReactionFromSmarts(s) for s in self.reaction_smirks]

    def _identify_new_bonds(self, molecule):
        """Identify attachment points in scaffold"""
        attachment_points = []
        match_idxs = molecule.GetSubstructMatch(self.scaffold)
        for idx in match_idxs:
            atom = molecule.GetAtomWithIdx(idx)
            neighbour_atoms = atom.GetNeighbors()
            for natom in neighbour_atoms:
                nidx = natom.GetIdx()
                if nidx not in match_idxs:
                    attachment_points.append((idx, nidx))
        return attachment_points

    def _get_synthons(self, molecule, reaction):
        """Apply a single reaction"""
        synthons = reaction.RunReactant(molecule, 0)
        return list(synthons)

    def _get_all_possible_synthons(self, molecule):
        """Apply all reactions"""
        synthons = []
        for reaction in self.reactions:
            synthons.extend(self._get_synthons(molecule, reaction))
        return synthons

    def _analyze_synthons(self, synthons, new_bonds):
        """Check to see if any synthons involved disconnecting the new bonds"""
        available_disconnections = [False] * len(new_bonds)

        for i, (aidx1, aidx2) in enumerate(new_bonds):
            for synth1, synth2 in synthons:
                synth1_idxs = set(
                    [
                        int(atom.GetProp("react_atom_idx"))
                        for atom in synth1.GetAtoms()
                        if atom.HasProp("react_atom_idx")
                    ]
                )
                synth2_idxs = set(
                    [
                        int(atom.GetProp("react_atom_idx"))
                        for atom in synth2.GetAtoms()
                        if atom.HasProp("react_atom_idx")
                    ]
                )
                # Stop as soon as we find a possible disconnection
                if (aidx1 in synth1_idxs) and (aidx2 in synth2_idxs):
                    available_disconnections[i] = True
                    break
                elif (aidx1 in synth2_idxs) and (aidx2 in synth1_idxs):
                    available_disconnections[i] = True
                    break
                else:
                    pass
        return available_disconnections

    def _score_smiles(self, smiles: str):
        """Score a single SMILES string"""
        molecule = get_mol(smiles)
        if molecule:
            new_bonds = self._identify_new_bonds(molecule)
            if not new_bonds:
                return 0.0
            synthons = self._get_all_possible_synthons(molecule)
            available_disconnections = self._analyze_synthons(synthons, new_bonds)
            return np.sum(available_disconnections) / len(available_disconnections)
        else:
            return 0.0

    def __call__(self, smiles, **kwargs):
        """Score a list of SMILES strings"""
        results = []
        scores = [score for score in self.mapper(self._score_smiles, smiles)]
        for smi, score in zip(smiles, scores):
            results.append({"smiles": smi, f"{self.prefix}_score": score})
        return results


class SelectiveDecoratedReactionFilter(DecoratedReactionFilter):
    def __init__(
        self,
        prefix: str,
        scaffold: str,
        allowed_reactions: dict = {},
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Name given to scoring function
        :param scaffold: Assumes de novo molecule generation is decorative from this scaffold with labeled attachment points
        :param allowed_reactions: A dictionary mapping attachment points to a list of allowed reactions e.g., {0: list(reaction)}}
        :param n_jobs: Number of parallel jobs to run
        """
        self.prefix = prefix.replace(" ", "_")
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.reaction_smirks = []
        self.scaffold = get_mol(scaffold)
        assert self.scaffold, f"Error parsing scaffold {scaffold}"
        assert (
            allowed_reactions
        ), "Please provide a dictionary of allowed reactions mapped to reaction vectors"
        assert all(
            [isinstance(reactions, list) for reactions in allowed_reactions.values()]
        ), "Reactions provided must be a list per reaction vector"
        self.reactions = {
            int(idx): [Chem.ReactionFromSmarts(s) for s in smirks]
            for idx, smirks in allowed_reactions.items()
        }

        # RDKit GetSubstructureMatch doesn't work with atom mapped molecules, so we need to remove the atom mapping and keep a record
        self.scaffidx_to_vector = {}
        for atom in self.scaffold.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                self.scaffidx_to_vector[atom.GetIdx()] = int(
                    atom.GetProp("molAtomMapNumber")
                )
        # Strip scaffold of atom mapping
        scaffold = re.sub(r"\[([a-zA-Z]):[0-9]\]", "\\1", scaffold)
        self.scaffold = get_mol(scaffold)
        assert self.scaffold, f"Error parsing scaffold {scaffold}"

    def _identify_new_bonds(self, molecule):
        """
        Identify attachment points in scaffold
        :returns: {Attachment point: (scaffold idx, new atom idx),}
        """
        attachment_points = {}
        match_idxs = molecule.GetSubstructMatch(self.scaffold)
        if not match_idxs:
            return attachment_points
        molidx_to_vector = {
            match_idxs[sidx]: vidx for sidx, vidx in self.scaffidx_to_vector.items()
        }
        for idx in match_idxs:
            atom = molecule.GetAtomWithIdx(idx)
            neighbour_atoms = atom.GetNeighbors()
            for natom in neighbour_atoms:
                nidx = natom.GetIdx()
                if (nidx not in match_idxs) and (idx in molidx_to_vector):
                    attachment_points[molidx_to_vector[idx]] = (idx, nidx)
        return attachment_points

    def _get_all_possible_synthons(self, molecule, vidx):
        """Apply all reactions"""
        synthons = []
        for reaction in self.reactions[vidx]:
            synthons.extend(self._get_synthons(molecule, reaction))
        return synthons

    def _analyze_synthons(self, synthons, new_bond):
        """Check to see if any synthons involved disconnecting the new bonds"""
        available_disconnections = False
        aidx1, aidx2 = new_bond

        for synth1, synth2 in synthons:
            synth1_idxs = set(
                [
                    int(atom.GetProp("react_atom_idx"))
                    for atom in synth1.GetAtoms()
                    if atom.HasProp("react_atom_idx")
                ]
            )
            synth2_idxs = set(
                [
                    int(atom.GetProp("react_atom_idx"))
                    for atom in synth2.GetAtoms()
                    if atom.HasProp("react_atom_idx")
                ]
            )
            # Stop as soon as we find a possible disconnection
            if (aidx1 in synth1_idxs) and (aidx2 in synth2_idxs):
                available_disconnections = True
                break
            elif (aidx1 in synth2_idxs) and (aidx2 in synth1_idxs):
                available_disconnections = True
                break
            else:
                pass
        return available_disconnections

    def _score_smiles(self, smiles: str):
        """Score a single SMILES string"""
        molecule = get_mol(smiles)
        if molecule:
            new_bonds = self._identify_new_bonds(molecule)
            if not new_bonds:
                return 0.0
            available_disconnections = []
            for vidx, bond in new_bonds.items():
                # Get specified synthons per attachment point
                synthons = self._get_all_possible_synthons(molecule, vidx)
                # Get available disconnections per attachment points
                available_disconnections.append(self._analyze_synthons(synthons, bond))
            return np.sum(available_disconnections) / len(available_disconnections)
        else:
            return 0.0
