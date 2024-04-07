import logging
import os
from typing import Union

from openeye import oechem, oeomega, oeshape
from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions.glide import GlideDock

logger = logging.getLogger("ROCS")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class ROCS:
    """
    Score structures based on shape alignment to a reference file.
    """

    return_metrics = [
        "GetColorScore",
        "GetColorTanimoto",
        "GetColorTversky",
        "GetComboScore",
        "GetFitColorTversky",
        "GetFitSelfColor",
        "GetFitSelfOverlap",
        "GetFitTversky",
        "GetFitTverskyCombo",
        "GetOverlap",
        "GetRefColorTversky",
        "GetRefSelfColor",
        "GetRefSelfOverlap",
        "GetRefTversky",
        "GetRefTverskyCombo",
        "GetShapeTanimoto",
        "GetTanimoto",
        "GetTanimotoCombo",
        "GetTversky",
        "GetTverskyCombo",
    ]

    def __init__(self, prefix: str, ref_file: Union[str, os.PathLike], **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., Risperidone)
        :param ref_file: Path to reference file to overlay query to (.pdb)
        :param kwargs:
        """

        self.prefix = prefix.replace(" ", "_")
        self.rocs_metrics = ROCS.return_metrics
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
        """Run ROCS on for query mol on ref mol -> rocs_result"""
        self.rocs_results = oeshape.OEROCSResult()
        oeshape.OEROCSOverlay(self.rocs_results, self.refmol, self.fitmol)
        return self

    def get_best_overlay(self):
        """Iterate over fitmol confs and get best fitting conformer -> best_overlay"""
        # Iterate through each conformer, assign result for best conformer
        for conf in self.fitmol.GetConfs():
            if conf.GetIdx() == self.rocs_results.GetFitConfIdx():
                self.rocs_results.Transform(conf)
                self.best_overlay = conf
        return self

    def __call__(
        self,
        smiles: list,
        directory: str,
        file_names: list,
        return_best_overlay: bool = False,
        **kwargs,
    ):
        """
        Calculate ROCS metrics for a list of smiles compared to a reference molecule.

        :param smiles: List of SMILES strings
        :param return_best_overlay: Whether to also return the best fitting conformer
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...] (, list best conformers)
        """
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_ROCS", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_names = file_names

        results = []
        best_overlays = []

        for smi, file in zip(smiles, self.file_names):
            result = {"smiles": smi}

            if Chem.MolFromSmiles(smi):
                try:
                    self.setup_smi(smi)
                    self.run_omega()

                    # Hack to catch 'valid molecules' that have no coordinates after omega initialisation
                    if len(self.fitmol.GetCoords()) == 0:
                        result.update(
                            {f"{self.prefix}_{m}": 0.0 for m in self.rocs_metrics}
                        )
                        continue

                    self.run_ROCS()
                    rocs_results = {
                        metric: getattr(self.rocs_results, metric)()
                        for metric in self.rocs_metrics
                    }
                    result.update(
                        {f"{self.prefix}_{m}": v for m, v in rocs_results.items()}
                    )
                    results.append(result)

                    # Best overlays
                    self.get_best_overlay()
                    best_overlays.append(self.best_overlay)
                    # Save to file
                    ofs = oechem.oemolostream()
                    if ofs.open(os.path.join(self.directory, f"{file}.sdf.gz")):
                        oechem.OEWriteMolecule(ofs, self.best_overlay)
                        result.update({f"{self.prefix}_best_conformer": file})

                except Exception:
                    logger.debug(f"{smi}: Can't process molecule")
                    result.update(
                        {f"{self.prefix}_{m}": 0.0 for m in self.rocs_metrics}
                    )
                    results.append(result)
            else:
                logger.debug(f"{smi}: rdkit molecule is None type")
                result.update({f"{self.prefix}_{m}": 0.0 for m in self.rocs_metrics})
                results.append(result)

        if return_best_overlay:
            assert len(results) == len(best_overlays)
            return results, best_overlays

        else:
            return results


class GlideDockFromROCS(GlideDock, ROCS):
    """
    Score structures based on Glide docking score with LigPrep ligand preparation,
     but using ROCS to align to a reference molecule and score in place
    """

    return_metrics = GlideDock.return_metrics + ROCS.return_metrics

    def __init__(
        self,
        prefix: str,
        glide_template: os.PathLike,
        ref_file: os.PathLike,
        cluster: str = None,
        timeout: float = 120.0,
        ligand_preparation: str = "epik",
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param glide_template: Path to a template docking file (.in)
        :param ref_file: Path to reference file to overlay query to (.pdb)
        :param cluster: Address to Dask scheduler for parallel processing via dask
        :param timeout: Timeout (seconds) before killing an individual docking simulation (only if using Dask for parallelisation)
        :param ligand_preparation: Whether to use 'ligprep' with limited default functionality, or 'epik' to protonate
        only the most probable state
        :param kwargs:
        """
        GlideDock.__init__(
            self,
            prefix=prefix,
            glide_template=glide_template,
            cluster=cluster,
            timeout=timeout,
            ligand_preparation=ligand_preparation,
        )
        ROCS.__init__(self, prefix=prefix, ref_file=ref_file)

        self.prefix = prefix.replace(" ", "")
        # Make sure glide template contains 'mininplace' method
        self.glide_options = self.modify_glide_in(
            self.glide_options, "DOCKING_METHOD", "mininplace"
        )

    def __call__(self, smiles: list, directory: str, file_names: list, **kwargs):
        """
        Calculate scores for GlideDockFromROCS based on a list of SMILES
        :param smiles: List of SMILES strings
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """

        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_GlideDock", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_names = file_names
        self.docking_results = []

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f"{step}_log.txt"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Refresh Dask every few hundred iterations
        if self.client is not None:
            if int(step) % 250 == 0:
                self.client.restart()

        # Prepare ligands
        self.prepare(smiles)

        # Read sdf files and run ROCS and write for docking
        results = []
        rocs_results = {}
        self.variants = {name: [] for name in self.file_names}
        for name in self.file_names:
            out_file = os.path.join(self.directory, f"{name}_ligprep.sdf")
            if os.path.exists(out_file):
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(
                    os.path.join(self.directory, f"{name}_ligprep.sdf")
                )
                for mol in supp:
                    if mol:
                        variant = mol.GetPropsAsDict()["s_lp_Variant"].split("-")[1]
                        self.variants[name].append(variant)
                        self.setup_smi(Chem.MolToSmiles(mol, isomericSmiles=True))
                        self.run_omega()

                        # Hack to catch 'valid molecules' that have no coordinates after omega init
                        if len(self.fitmol.GetCoords()) == 0:
                            rocs_results[f"{name}_{variant}"] = {
                                f"{self.prefix}_{m}": 0.0 for m in self.rocs_metrics
                            }
                            continue

                        # Run ROCS and write each variants best overlay
                        self.run_ROCS()
                        rocs_results[f"{name}_{variant}"] = {
                            m: getattr(self.rocs_results, m)()
                            for m in self.rocs_metrics
                        }
                        self.get_best_overlay()
                        ofs = oechem.oemolostream(
                            os.path.join(
                                self.directory, f"{name}-{variant}_ligprep.sdf"
                            )
                        )
                        if oechem.OEAddExplicitHydrogens(self.best_overlay):
                            oechem.OEWriteMolecule(ofs, self.best_overlay)
                        ofs.close()
                        logger.debug(f"Split and aligned {name} -> {name}-{variant}")

        self.run_glide()
        best_variants = self.get_docking_scores(smiles, return_best_variant=True)
        for result, best_variant in zip(self.docking_results, best_variants):
            result.update(rocs_results[best_variant])
            results.append(result)

        # Cleanup
        self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None
        self.fitmol = None
        self.rocs_results = None
        self.best_overlay = None
        self.docking_results = None

        # Check
        assert len(smiles) == len(results)
        return results
