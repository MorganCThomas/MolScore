from functools import partial
from itertools import combinations
from typing import Union

from rdkit.Chem import QED, Crippen, Descriptors, GetDistanceMatrix, GraphDescriptors
from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions.SA_Score import sascorer
from molscore.scoring_functions.utils import Pool, get_mol


class MolecularDescriptors:
    """
    Calculate a suite of molecular descriptors
    """

    return_metrics = [
        "QED",
        "SAscore",
        "CLogP",
        "MolWt",
        "HeavyAtomCount",
        "HeavyAtomMolWt",
        "NumHAcceptors",
        "NumHDonors",
        "NumHeteroatoms",
        "NumRotatableBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "RingCount",
        "TPSA",
        "PenLogP",
        "FormalCharge",
        "MolecularFormula",
        "Bertz",
        "MaxConsecutiveRotatableBonds",
        "FlourineCount",
    ]

    def __init__(self, prefix: str = "desc", n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., desc)
        :param n_jobs: Number of cores for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.strip().replace(" ", "_")
        self.results = None
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)

    @staticmethod
    def calculate_descriptors(smi, prefix):
        descriptors = {
            "QED": QED.qed,
            "SAscore": sascorer.calculateScore,
            "CLogP": Crippen.MolLogP,
            "MolWt": Descriptors.MolWt,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHeteroatoms": Descriptors.NumHeteroatoms,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "NumAliphaticRings": Descriptors.NumAliphaticRings,
            "RingCount": Descriptors.RingCount,
            "TPSA": Descriptors.TPSA,
            "PenLogP": MolecularDescriptors.penalized_logp,
            "FormalCharge": Chem.GetFormalCharge,
            "MolecularFormula": Descriptors.rdMolDescriptors.CalcMolFormula,
            "Bertz": GraphDescriptors.BertzCT,
            "MaxConsecutiveRotatableBonds": MolecularDescriptors.max_consecutive_rotatable_bonds,
            "FlourineCount": MolecularDescriptors.flourine_count,
        }
        descriptors = {f"{prefix}_{k}": v for k, v in descriptors.items()}

        result = {"smiles": smi}
        mol = get_mol(smi)
        if mol is not None:
            for k, v in descriptors.items():
                try:
                    result.update({k: v(mol)})
                # If any error is thrown append 0.0.
                except Exception:
                    result.update({k: 0.0})
        else:
            result.update({k: 0.0 for k in descriptors.keys()})
        return result

    @staticmethod
    def penalized_logp(mol: Chem.rdchem.Mol):
        """Calculates the penalized logP of a molecule.
        Refactored from
        https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
        See Junction Tree Variational Autoencoder for Molecular Graph Generation
        https://arxiv.org/pdf/1802.04364.pdf

        Section 3.2
        Penalized logP is defined as:
         y(m) = logP(m) - SA(m) - cycle(m)
         y(m) is the penalized logP,
         logP(m) is the logP of a molecule,
         SA(m) is the synthetic accessibility score,
         cycle(m) is the largest ring size minus by six in the molecule.

         :param mol: rdkit mol
         :return Penalized LogP
        """
        # Get largest cycle length
        cycle_list = mol.GetRingInfo().AtomRings()
        if cycle_list:
            cycle_length = max([len(j) for j in cycle_list])
        else:
            cycle_length = 0

        log_p = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
        cycle_score = max(cycle_length - 6, 0)
        return log_p - sa_score - cycle_score

    @staticmethod
    def consecutive_rotatable_bonds(
        mol: Chem.rdchem.Mol, include_ring_connections: bool = True
    ):
        """
        Calculate the consecutive rotatable bonds in a molecule, correcting for amides and esters.
        :param mol: Molecule as SMILES or RDKit Mol
        :param include_ring_connections: Whether to include ring connections (will only be of length 1).
        :return: List with rotatable bonds grouped by conesecutive connections
        """
        rotatable_chains = []
        # !D1 More than one bond (exclude terminal atoms)
        # - Single alihpatic bond between two atoms
        # At least one of them isn't in a ring
        rb_patt = Chem.MolFromSmarts("[*!R!D1]-[*!D1]")
        amide_patt = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        ester_patt = Chem.MolFromSmarts("[OX2][CX3](=[OX1])")

        rb_matches = mol.GetSubstructMatches(rb_patt)
        amide_matches = mol.GetSubstructMatches(amide_patt)
        ester_matches = mol.GetSubstructMatches(ester_patt)

        for ai, aj in rb_matches:
            # Correct for amides & Esters
            if any([(ai in amide) and (aj in amide) for amide in amide_matches]):
                continue
            if any([(ai in ester) and (aj in ester) for ester in ester_matches]):
                continue

            # If either atom found in a set add bonded atom to that set
            for chain in rotatable_chains:
                if ai in chain:
                    chain.add(aj)
                    break
                elif aj in chain:
                    chain.add(ai)
                    break
                else:
                    pass

            # Both atoms aren't in any chain, add as a new set
            if all([ai not in chain for chain in rotatable_chains]) and all(
                [aj not in chain for chain in rotatable_chains]
            ):
                rotatable_chains.append(set([ai, aj]))

        if include_ring_connections:
            # Single bond between two ring atoms
            rb_rc_patt = Chem.MolFromSmarts("[R!D1]-[R!D1]")
            rb_rc_matches = mol.GetSubstructMatches(rb_rc_patt)
            ring_info = mol.GetRingInfo()

            for ri, rj in rb_rc_matches:
                # If they're both in the same ring ignore
                if any(
                    [(ri in ring) and (rj in ring) for ring in ring_info.AtomRings()]
                ):
                    continue
                else:
                    rotatable_chains.append(set([ri, rj]))

        return rotatable_chains

    @classmethod
    def max_consecutive_rotatable_bonds(
        cls, mol: Union[str, Chem.rdchem.Mol], include_ring_connections: bool = True
    ):
        """
        Calculate the consecutive rotatable bonds in a molecule, correcting for amides and esters.
        :param mol: Molecule as SMILES or RDKit Mol
        :param include_ring_connections: Whether to include ring connections (will only be of length 1).
        :return: List with rotatable bonds grouped by conesecutive connections
        """
        # Check Mol
        mol = get_mol(mol)
        if mol is None:
            return 0

        rotatable_chains = cls.consecutive_rotatable_bonds(
            mol=mol, include_ring_connections=include_ring_connections
        )

        try:
            max_chain_length = (
                len(sorted(rotatable_chains, key=lambda x: len(x))[-1]) - 1
            )
        except IndexError:
            max_chain_length = 0
        return max_chain_length

    @staticmethod
    def charge_counts(mol: Union[str, Chem.rdchem.Mol]):
        """
        Count the charges based on SMILES, correct for valence separated charges e.g., Nitro
        :param mol: An rdkit mol or str
        :return: Net charge, positive charge, negative charge
        """
        # SMARTS pattern to find single charges on atoms
        charge_pattern = Chem.MolFromSmarts(
            "[+1!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
        )

        # Accept smiles / mol
        if mol:
            mol = get_mol(mol)
        if mol is None:
            return 0, 0, 0

        # Count charges
        positive_charge = 0
        negative_charge = 0
        at_matches = mol.GetSubstructMatches(charge_pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                if chg > 0:
                    positive_charge += chg
                else:
                    negative_charge += chg

        net_charge = positive_charge + negative_charge

        return net_charge, positive_charge, negative_charge

    @staticmethod
    def flourine_count(mol: Union[str, Chem.rdchem.Mol]):
        """
        Count the number of flourines in a Molecule
        :param mol: An rdkit mol or str
        :return: #Flourines
        """
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate the scores for RDKitDescriptors
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        pcalculate_descriptors = partial(
            self.calculate_descriptors, prefix=self.prefix
        )
        results = [result for result in self.mapper(pcalculate_descriptors, smiles)]

        return results


class LinkerDescriptors(MolecularDescriptors):
    """
    Calculate a linker focussed molecular descriptors
    """

    return_metrics = [
        "EffectiveLength",
        "MaxLength",
        "LengthRatio",
        "RingCount",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumHetatoms",
        "NumSP",
        "NumSP2",
        "NumSP3",
        "NumHDonors",
        "NumHAcceptors",
        "MolWt",
        "HeavyAtomCount",
        "RatioRotatableBonds",
        "MaxConsecutiveRotatableBonds",
    ]

    def __init__(self, prefix: str = "linker_desc", n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., desc)
        :param n_jobs: Number of cores for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.strip().replace(" ", "_")
        self.n_jobs = n_jobs

    @staticmethod
    def _strip_attachment_points(smiles: str):
        smiles = smiles.replace("(*)", "").replace("*", "")
        return smiles

    @staticmethod
    def _linker_rotatable_bonds(lmol, max_consecutive: bool = False):
        if max_consecutive:
            rotatable_bonds = MolecularDescriptors.max_consecutive_rotatable_bonds(lmol)
        else:
            rotatable_bonds = Descriptors.NumRotatableBonds(lmol)

        # Add correction assuming bond to fragment is rotatable if not a ring connection
        correction = 0
        if lmol.GetNumBonds() > 1:
            for atom in lmol.GetAtoms():
                if (atom.GetSymbol() == "*") and not (atom.IsInRing()):
                    correction += 1
        else:
            bond = lmol.GetBonds()[0]
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                correction = 1

        return rotatable_bonds + correction

    def _score(self, linker: str):
        descs = {}
        if not linker:
            return {m: 0.0 for m in self.return_metrics}

        lmol = Chem.MolFromSmiles(linker)

        if not lmol:
            return {m: 0.0 for m in self.return_metrics}

        # Get attachment point indices
        at_pts = []
        for atom in lmol.GetAtoms():
            if atom.GetSymbol() == "*":
                at_pts.append(atom.GetIdx())
        distances = GetDistanceMatrix(lmol)
        # Calculate effective length
        effective_lengths = []
        for idx1, idx2 in combinations(at_pts, 2):
            effective_lengths.append(distances[idx1, idx2])
        if effective_lengths:
            effective_length = min(effective_lengths)
        else:
            effective_length = 0.0
        descs["EffectiveLength"] = int(effective_length)
        # Calculate max length
        max_length = distances.max()
        descs["MaxLength"] = int(max_length)
        # Calculate ratio
        descs["LengthRatio"] = effective_length / max_length
        # Calculate # rings
        descs["RingCount"] = Descriptors.RingCount(lmol)
        # Calculate # aromatic rings
        descs["NumAromaticRings"] = Descriptors.NumAromaticRings(lmol)
        # Calculate # aliphatic rings
        descs["NumAliphaticRings"] = Descriptors.NumAliphaticRings(lmol)
        # Calculate # heterotoms
        descs["NumHetatoms"] = Descriptors.NumHeteroatoms(lmol) - linker.count("*")
        # Calculate # sp atoms
        descs["NumSP"] = len(
            [
                atom
                for atom in lmol.GetAtoms()
                if atom.GetHybridization() == Chem.HybridizationType.SP
            ]
        )
        descs["NumSP2"] = len(
            [
                atom
                for atom in lmol.GetAtoms()
                if atom.GetHybridization() == Chem.HybridizationType.SP2
            ]
        )
        descs["NumSP3"] = len(
            [
                atom
                for atom in lmol.GetAtoms()
                if atom.GetHybridization() == Chem.HybridizationType.SP3
            ]
        )
        # Calculate # hbd/hba
        descs["NumHDonors"] = Descriptors.NumHDonors(lmol)
        descs["NumHAcceptors"] = Descriptors.NumHAcceptors(lmol)
        # Calculate MolWt / HA
        descs["MolWt"] = Descriptors.MolWt(lmol)
        descs["HeavyAtomCount"] = Descriptors.HeavyAtomCount(lmol)
        # Calculate Ratio of rotatable bonds
        if lmol.GetNumBonds() > 0:
            descs["NumRotatableBonds"] = self._linker_rotatable_bonds(lmol)
            descs["RatioRotatableBonds"] = (
                descs["NumRotatableBonds"] / lmol.GetNumBonds()
            )
            descs["MaxConsecutiveRotatableBonds"] = self._linker_rotatable_bonds(
                lmol, max_consecutive=True
            )
        else:
            descs["NumRotatableBonds"] = 0.0
            descs["RatioRotatableBonds"] = 0.0
            descs["MaxConsecutiveRotatableBonds"] = 0.0

        return descs

    def __call__(self, smiles: list, additional_formats: dict, **kwargs):
        """
        Calculate the scores for RDKitDescriptors
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        assert additional_formats and (
            "linker" in additional_formats.keys()
        ), "LinkerDescriptors requires a linker format in additional_formats"
        results = [
            {"smiles": smi, "linker": linker}
            for smi, linker in zip(smiles, additional_formats["linker"])
        ]

        # Compute descriptors in parallel
        descs = [r for r in self.mapper(self._score, additional_formats["linker"])]
        # Add prefix
        descs = [{f"{self.prefix}_{k}": v for k, v in ds.items()} for ds in descs]

        for r, d in zip(results, descs):
            r.update(d)

        return results


# Backwards compatability with old config files as it used to be called RDKit descriptors
class RDKitDescriptors(MolecularDescriptors):
    """
    Calculate a suite of molecular descriptors
    """

    def __init__(self, prefix: str = "desc", n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., desc)
        :param n_jobs: Number of cores for multiprocessing
        :param kwargs:
        """
        super().__init__(prefix=prefix, n_jobs=n_jobs, **kwargs)
