from multiprocessing import Pool
from functools import partial
from rdkit.Chem import Descriptors, QED, Crippen, GraphDescriptors
from rdkit.Chem import AllChem as Chem
from molscore.scoring_functions.SA_Score import sascorer
from molscore.scoring_functions.utils import charge_counts, max_consecutive_rotatable_bonds


class RDKitDescriptors:
    """
    Score structures based rdkit descriptors
    """

    return_metrics = ['QED', 'SAscore', 'CLogP', 'MolWt', 'HeavyAtomCount', 'HeavyAtomMolWt',
                      'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
                      'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'TPSA', 'PenLogP',
                      'FormalCharge', 'MolecularFormula', 'Bertz', 'MaxConsecutiveRotatableBonds']

    def __init__(self, prefix: str = 'desc', n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., desc)
        :param n_jobs: Number of cores for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.strip().replace(' ', '_')
        self.results = None
        self.n_jobs = n_jobs

    @staticmethod
    def calculate_descriptors(smi, prefix):

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

        descriptors = {'QED': QED.qed,
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
                       'PenLogP': penalized_logp,
                       'FormalCharge': Chem.GetFormalCharge,
                       'MolecularFormula': Descriptors.rdMolDescriptors.CalcMolFormula,
                       'Bertz': GraphDescriptors.BertzCT}
        descriptors = {f'{prefix}_{k}': v for k, v in descriptors.items()}

        result = {'smiles': smi}
        mol = Chem.MolFromSmiles(smi)
        if mol:
            for k, v in descriptors.items():
                try:
                    result.update({k: v(mol)})
                # If any error is thrown append 0.0.
                except:
                    result.update({k: 0.0})

            # Add custom descriptors
            result.update({f'{prefix}_MaxConsecutiveRotatableBonds': max_consecutive_rotatable_bonds(mol)})

        else:
            result.update({k: 0.0 for k in descriptors.keys()})
            # Add custom descriptors
            result.update({f'{prefix}_MaxConsecutiveRotatableBonds': 0.0})
        return result

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate the scores for RDKitDescriptors
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        with Pool(self.n_jobs) as pool:
            pcalculate_descriptors = partial(self.calculate_descriptors, prefix=self.prefix)
            results = [result for result in pool.imap(pcalculate_descriptors, smiles)]

        return results
