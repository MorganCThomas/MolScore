__all__ = ["MockGenerator", "BaseTests", "test_files"]

import os

from molscore import MockGenerator
from molscore.tests.base_tests import BaseTests

test_files = {
    "GlideDock_grid": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "DRD2_6CM4_grid.zip"
    ),
    "DRD2_receptor": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "DRD2_6CM4p_rec.pdb"
    ),
    "DRD2_receptor_pdbqt": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "DRD2_6CM4p_rec.pdbqt"
    ),
    "DRD2_ref_ligand": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "DRD2_risperidone.sdf"
    ),
    "DRD2_ref_smiles": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2O",
    "DRD2_rdock_constraint": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "DRD2_3x32.restr"
    ),
    "chemprop_model": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "chemprop_model"
    ),
}
