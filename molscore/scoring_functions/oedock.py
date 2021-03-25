"""
This code is adapted from reinvent-memory
https://github.com/tblaschke/reinvent-memory
"""

import os
import logging
import openeye.oechem as oechem
import openeye.oedocking as oedocking
import openeye.oeomega as oeomega

logger = logging.getLogger('oedock')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class FRED:
    """
    Score structures using OpenEye docking (FRED)
    """
    return_metrics = ['FRED_energy']

    def __init__(self, prefix: str, receptor_file: os.PathLike):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., Risperidone)
        :param receptor_file: Path to receptor file (.oeb).
        """
        logger.warning("This code has not been tested (at all!)")
        self.prefix = prefix
        self.receptor_file = receptor_file
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        self.omega = oeomega.OEOmega(omegaOpts)
        oechem.OEThrow.SetLevel(10000)
        oereceptor = oechem.OEGraphMol()
        oedocking.OEReadReceptorFile(oereceptor, self.receptor_file)
        self.oedock = oedocking.OEDock()
        self.oedock.Initialize(oereceptor)

    @staticmethod
    def dock():
        # TODO check parallel implementation of this? can't test ... don't have license
        raise NotImplementedError

    def __call__(self, smiles: list):
        """
        Calculate the scores for FRED given a list of SMILES strings.
        :param smiles: List of SMILES strings.
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = []
        for smi in smiles:
            result = {'smiles': smi}
            mol = oechem.OEMol()
            if not oechem.OESmilesToMol(mol, smi):
                results.append({'smiles': smi, f'{self.prefix}_FRED_energy': 0.0})
                continue
            if self.omega(mol):
                dockedMol = oechem.OEGraphMol()
                self.oedock.DockMultiConformerMolecule(dockedMol, mol)
                score = dockedMol.GetEnergy()
                results.append({'smiles': smi, f'{self.prefix}_FRED_energy': score})
        return results
