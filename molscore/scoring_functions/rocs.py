import logging

from rdkit.Chem import AllChem as Chem

from openeye import oechem
from openeye import oeomega
from openeye import oeshape

logger = logging.getLogger('ROCS')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class ROCS:
    """
    Shape alignment on generated molecules. 
    """
    def __init__(self, prefix: str, ref_file: str, **kwargs):
        """
        Shape alignment on generated molecules to a reference molecule
        :param prefix: Name (to help keep track metrics, if using a scoring function class more than once)
        :param ref_file: Path to reference file to overlay query to (.pdb)
        :param return_best_overlay: Whether to also return best overlay (for use with docking)
        :param kwargs: Ignored
        """

        self.prefix = prefix.replace(" ", "_")
        self.rocs_metrics = ['GetColorScore', 'GetColorTanimoto', 'GetColorTversky', 'GetComboScore',
                             'GetFitColorTversky', 'GetFitSelfColor', 'GetFitSelfOverlap', 'GetFitTversky',
                             'GetFitTverskyCombo', 'GetOverlap', 'GetRefColorTversky', 'GetRefSelfColor',
                             'GetRefSelfOverlap', 'GetRefTversky', 'GetRefTverskyCombo', 'GetShapeTanimoto',
                             'GetTanimoto', 'GetTanimotoCombo', 'GetTversky', 'GetTverskyCombo']
        self.ref_file = ref_file
        self.refmol = oechem.OEMol()
        ifs = oechem.oemolistream(self.ref_file)  # Set up input file stream
        oechem.OEReadMolecule(ifs, self.refmol)  # Read ifs to empty mol object
        self.fitmol = None
        self.rocs_results = None
        self.best_overlay = None

        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetMaxSearchTime(1.0)
        self.omega = oeomega.OEOmega(omegaOpts)

    def setup_smi(self, smiles: str):
        """
        Load SMILES string into OE mol object
        :param smiles: SMILES string
        :return:
        """
        self.fitmol = oechem.OEMol()
        oechem.OESmilesToMol(self.fitmol, smiles)

        return self

    def run_omega(self):
        """Run omega on query mol"""
        self.omega(self.fitmol)
        return self

    def run_ROCS(self):
        """ Run ROCS on for query mol on ref mol -> rocs_result"""
        self.rocs_results = oeshape.OEROCSResult()
        oeshape.OEROCSOverlay(self.rocs_results, self.refmol, self.fitmol)
        return self

    def get_best_overlay(self):
        """ Iterate over fitmol confs and get best fitting conformer -> best_overlay"""
        # Iterate through each conformer, assign result for best conformer
        for conf in self.fitmol.GetConfs():
            if conf.GetIdx() == self.rocs_results.GetFitConfIdx():
                self.rocs_results.Transform(conf)
                self.best_overlay = conf
        return self

    def __call__(self, smiles: list, return_best_overlay: bool = False, **kwargs):
        """
        Calculate ROCS metrics for a list of smiles compared to a reference molecule.

        :param smiles: List of SMILES strings
        :param return_best_overlay: Whether to also return the best fitting conformer
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...] (, list best conformers)
        """
        results = []
        best_overlays = []

        for smi in smiles:
            result = {'smiles': smi}

            if Chem.MolFromsmiles(smi):
                try:
                    self.setup_smi(smi)
                    self.run_omega()

                    # Hack to catch 'valid molecules' that have no coordinates after omega initialisation
                    if len(self.fitmol.GetCoords()) == 0:
                        result.update({f'{self.prefix}_{m}': 0.0 for m in self.rocs_metrics})
                        continue

                    self.run_ROCS()
                    rocs_results = {metric: getattr(self.rocs_results, metric)() for metric in self.rocs_metrics}
                    result.update({f'{self.prefix}_{m}': v for m, v in rocs_results.items()})
                    results.append(result)

                    if return_best_overlay:
                        self.get_best_overlay()
                        best_overlays.append(self.best_overlay)

                except:
                    logger.debug(f'{smi}: Can\'t process molecule')
                    result.update({f'{self.prefix}_{m}': 0.0 for m in self.rocs_metrics})
                    results.append(result)
            else:
                logger.debug(f'{smi}: rdkit molecule is None type')
                result.update({f'{self.prefix}_{m}': 0.0 for m in self.rocs_metrics})
                results.append(result)

        if return_best_overlay:
            assert len(results) == len(best_overlays)
            return results, best_overlays

        else:
            return results
