from rdkit.Chem import Descriptors, QED, Crippen
from rdkit.Chem import AllChem as Chem
from molscore.scoring_functions.SA_Score import sascorer


class RDKitDescriptors:
    """
    Score structures based rdkit descriptors
    """

    return_metrics = ['QED', 'SAscore', 'CLogP', 'MolWt', 'HeavyAtomCount', 'HeavyAtomMolWt',
                      'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
                      'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'TPSA', 'PenLogP', 'FormalCharge']

    def __init__(self, prefix: str = 'desc', **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., desc)
        :param kwargs:
        """
        self.prefix = prefix.strip().replace(' ', '_')
        self.results = None
        self.descriptors = {'QED': QED.qed,
                            'SAscore': sascorer.calculateScore,
                            'CLogP': Crippen.MolLogP,
                            'MolWt': Descriptors.MolWt,
                            'HeavyAtomCount': Descriptors.HeavyAtomCount,
                            'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
                            'NumHAcceptors': Descriptors.NumHAcceptors,
                            'NumHDonors': Descriptors.NumHDonors,
                            'NumHeteroatoms': Descriptors.NumHeteroatoms,
                            'NumRotatableBonds': Descriptors.NumRotatableBonds,
                            'NumAromaticRings': Descriptors.NumAromaticRings,
                            'NumAliphaticRings': Descriptors.NumAliphaticRings,
                            'RingCount': Descriptors.RingCount,
                            'TPSA': Descriptors.TPSA,
                            'PenLogP': self.penalized_logp,
                            'FormalCharge': Chem.GetFormalCharge}
        # Add prefix
        self.descriptors = {f'{self.prefix}_{k}': v for k, v in self.descriptors.items()}


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
