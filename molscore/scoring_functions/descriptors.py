from rdkit.Chem import Descriptors, QED, Crippen
from rdkit.Chem import AllChem as Chem
from molscore.scoring_functions.SA_Score import sascorer


class RDKitDescriptors:
    """
    Scoring function class to grab a variety of descriptors from rdkit.
    """
    def __init__(self, **kwargs):
        """
        Scoring function class to grab a variety of descriptors from rdkit.
        :param kwargs: Ignored
        """
        self.results = None
        self.descriptors = {'desc_QED': QED.qed,
                            'desc_SAscore': sascorer.calculateScore,
                            'desc_CLogP': Crippen.MolLogP,
                            'desc_MolWt': Descriptors.MolWt,
                            'desc_HeavyAtomCount': Descriptors.HeavyAtomCount,
                            'desc_HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
                            'desc_NumHAcceptors': Descriptors.NumHAcceptors,
                            'desc_NumHDonors': Descriptors.NumHDonors,
                            'desc_NumHeteroatoms': Descriptors.NumHeteroatoms,
                            'desc_NumRotatableBonds': Descriptors.NumRotatableBonds,
                            'desc_NumAromaticRings': Descriptors.NumAromaticRings,
                            'desc_NumAliphaticRings': Descriptors.NumAliphaticRings,
                            'desc_RingCount': Descriptors.RingCount,
                            'desc_TPSA': Descriptors.TPSA,
                            'desc_PenLogP': self.penalized_logp,
                            'desc_FormalCharge': Chem.GetFormalCharge}

    def get_largest_ring_size(self, mol: Chem.rdchem.Mol):
        """
        Calculates the largest ring size of a molecule.
        Refactored from
        https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
        :param mol: rdkit mol
        :return Cycle
        """
        cycle_list = mol.GetRingInfo().AtomRings()
        if cycle_list:
            cycle_length = max([len(j) for j in cycle_list])
        else:
            cycle_length = 0
        return cycle_length

    def penalized_logp(self, mol: Chem.rdchem.Mol):
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
        log_p = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
        largest_ring_size = self.get_largest_ring_size(mol)
        cycle_score = max(largest_ring_size - 6, 0)
        return log_p - sa_score - cycle_score

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate the scores for RDKitDescriptors
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = []
        for smi in smiles:
            result = {'smiles': smi}
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for k, v in self.descriptors.items():
                    try:
                        result.update({k: v(mol)})
                    # If any error is thrown append 0.0.
                    except:
                        result.update({k: 0.0})
            else:
                result.update({k: 0.0 for k in self.descriptors.keys()})
            results.append(result)

        return results
