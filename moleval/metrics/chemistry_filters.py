from moleval.metrics.metrics_utils import get_mol, mapper, mol_passes_filters

class ChemistryFilter():
    def __init__(self, target: list = None, n_jobs=1):
        self.n_jobs = n_jobs
        # Preprocess target
        # MolWt, LogP, SillyWalks?, Does this make sense, what if the target dataset is very broad anyway

    @staticmethod
    def passes_basic(mol):
        passes = mol_passes_filters(
            mol=mol,
            allowed=None, # Means allowed atoms are {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
            allow_charge=True,
            isomericSmiles=False,
            molwt_min=150,
            molwt_max=650,
            mollogp_max=4.5,
            rotatable_bonds_max=7,
            filters=True # MOSES MCF and PAINS filters
            )
        return passes

    def passes_target(self, mol):
        raise NotImplementedError

    def filter_molecule(self, mol, basic=True, target=False):
        mol = get_mol(mol)
        passes = False
        if mol:
            if basic:
                passes = self.passes_basic(mol)
            if target:
                passes = self.passes_target(mol)
        return passes
    
    def filter_molecules(self, mols, basic=True, target=False):
        results = mapper(self.n_jobs)(self.filter_molecule, mols)
        return results