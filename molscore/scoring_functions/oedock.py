import atexit
import glob
import logging
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Union

from openeye import oechem, oedocking, oeomega

from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import DaskUtils

logger = logging.getLogger("oedock")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class OEDock:
    """
    Score structures using OpenEye docking (FRED)
    """

    return_metrics = [
        "docking_score",
        "NetCharge",
        "PositiveCharge",
        "NegativeCharge",
        "best_variant",
    ]

    def __init__(
        self,
        prefix: str,
        receptor: Union[str, os.PathLike],
        ref_ligand: Union[str, os.PathLike],
        ligand_preparation: str,
        prep_timeout: float = 30.0,
        omega_energy_window: int = 10,
        omega_max_confs: int = 200,
        omega_strict_stereo: bool = False,
        dock_scoring_function: str = "Chemgauss4",
        dock_search_resolution: str = "Standard",
        dock_num_poses: int = 1,
        cluster: Union[str, int] = None,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., Risperidone)
        :param receptor: Path to receptor file (.oeb, .oez, .pdb, .mol2, .mmcif)
        :param ref_ligand: Path to reference ligand file used to determine binding site (.sdf, .mol2, .pdb)
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param omega_energy_window: Sets the maximum allowable energy difference between the lowest and the highest energy conformers, in units of kcal/mol
        :param omega_max_confs: Sets the maximum number of conformers to be kept
        :param omega_strict_stereo: Sets whether conformer generation should fail if stereo is not specified on the input molecules
        :param dock_scoring_function: Docking scoring function used for search and optimization [Shapegauss, PLP, Chemgauss3, Chemgauss4, Chemscore, Hybrid1, Hybrid]
        :param dock_search_resolution: Rotational/translational RMSD step sizes used during search and optimization [High, Standard, Low]
        :param dock_num_poses: Number of poses for docking to return
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        """
        self.prefix = prefix.replace(" ", "_")
        self.prep_timeout = prep_timeout
        if "timeout" in kwargs.items():
            self.prep_timeout = float(kwargs["timeout"])  # Back compatability
        self.temp_dir = TemporaryDirectory()
        self.directory = None
        self.file_names = None
        self.variants = None
        self.mols = None

        # Load receptor
        rec_ifs = oechem.oemolistream(os.path.abspath(receptor))
        rec_ext = os.path.basename(receptor).split(".")[-1]
        assert rec_ext in ["pdb", "mmcif", "mol2", "oeb", "oez"]
        rec_ifs.SetFormat(getattr(oechem, f"OEFormat_{rec_ext.upper()}"))
        self.receptor = oechem.OEGraphMol()
        oechem.OEReadMolecule(rec_ifs, self.receptor)

        # Load ref ligand
        lig_ifs = oechem.oemolistream(os.path.abspath(ref_ligand))
        lig_ext = os.path.basename(ref_ligand).split(".")[-1]
        assert lig_ext in ["sdf", "mol2", "pdb"]
        lig_ifs.SetFormat(getattr(oechem, f"OEFormat_{lig_ext.upper()}"))
        self.ref_lig = oechem.OEGraphMol()
        oechem.OEReadMolecule(lig_ifs, self.ref_lig)

        # Prepare complex
        self.complex = self.receptor.CreateCopy()
        amap, bmap = oechem.OEAddMols(self.complex, self.ref_lig)

        # Make OEReceptor needed for docking
        oedocking.OEMakeReceptor(self.receptor, self.complex, self.ref_lig)

        # Omega options
        self.omega_energy_window = omega_energy_window
        self.omega_max_confs = omega_max_confs
        self.omega_strict_stereo = omega_strict_stereo

        # Docking options
        self.dock_scoring_function = dock_scoring_function
        self.dock_search_resolution = dock_search_resolution
        self.dock_num_poses = dock_num_poses

        # Setup dask
        self.client = DaskUtils.setup_dask(
            cluster_address_or_n_workers=cluster,
            local_directory=self.temp_dir.name,
            logger=logger,
        )
        atexit.register(DaskUtils._close_dask, self.client)

        # Select ligand preparation protocol
        self.ligand_protocol = [
            p
            for p in ligand_preparation_protocols
            if ligand_preparation.lower() == p.__name__.lower()
        ][0]  # Back compatible
        if self.client is not None:
            self.ligand_protocol = self.ligand_protocol(
                dask_client=self.client, timeout=self.prep_timeout, logger=logger
            )
        else:
            self.ligand_protocol = self.ligand_protocol(logger=logger)

    def load_prepared_ligand(self) -> oechem.OEMol:
        """
        Load a prepared ligand .sdf file and assign the name and variant to it's generic data
        """
        mols = []
        for name in self.variants:
            for variant in self.variants[name]:
                file_path = os.path.join(
                    self.directory, f"{name}-{variant}_prepared.sdf"
                )
                if os.path.exists(file_path):
                    logger.debug(f"Found {name}-{variant}")
                    ifs = oechem.oemolistream(file_path)
                    ifs.SetFormat(oechem.OEFormat_SDF)
                    mol = oechem.OEMol()
                    if oechem.OEReadMolecule(ifs, mol):
                        # Add variant to mol for tracking
                        mol.AddData(oechem.OEGetTag("name"), name)
                        mol.AddData(oechem.OEGetTag("variant"), variant)
                        mols.append(mol)
                    else:
                        logger.debug(f"Unable to read {name}-{variant}_prepared.sdf")
                else:
                    logger.debug(f"Unable to find {name}-{variant}_preapred.sdf")
        return mols

    @staticmethod
    def omega(
        mol: oechem.OEMol, omega_energy_window, omega_max_confs, omega_strict_stereo
    ) -> oechem.OEMol:
        """
        Generate conformers using omega
        """
        # Setup omega in thread (object not pickle-able)
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetEnergyWindow(omega_energy_window)
        omegaOpts.SetMaxConfs(omega_max_confs)
        omegaOpts.SetStrictStereo(omega_strict_stereo)
        omega = oeomega.OEOmega(omegaOpts)
        # Run omega
        omega(mol)
        return mol

    def run_omega(self):
        """
        Run omega in parallel if cluster specified
        """
        # Initialize subprocess
        logger.debug("Omega called")
        pomega = partial(
            self.omega,
            omega_energy_window=self.omega_energy_window,
            omega_max_confs=self.omega_max_confs,
            omega_strict_stereo=self.omega_strict_stereo,
        )

        if self.client is not None:
            futures = self.client.map(pomega, self.mols)
            self.mols = self.client.gather(futures)
        else:
            self.mols = [pomega(m) for m in self.mols]
        logger.debug("Omega finished")
        return

    @staticmethod
    def dock(
        mol: oechem.OEMol,
        receptor,
        dock_scoring_function,
        dock_search_resolution,
        num_poses,
    ) -> oechem.OEMol:
        """
        Docking multi conformer mols using oedock
        """
        # Setup dock in thread (object not pickle-able)
        dockOpts = oedocking.OEDockOptions()
        dockOpts.SetScoreMethod(
            getattr(oedocking, f"OEDockMethod_{dock_scoring_function}")
        )
        dockOpts.SetResolution(
            getattr(oedocking, f"OESearchResolution_{dock_search_resolution}")
        )
        oedock = oedocking.OEDock(dockOpts)
        oedock.Initialize(receptor)
        # Run dock
        dockedMol = oechem.OEMol()
        oedock.DockMultiConformerMolecule(dockedMol, mol, num_poses)
        # Save data
        dockedMol.AddData(
            oechem.OEGetTag("name"), mol.GetData("name")
        )  # Copy over name
        dockedMol.AddData(
            oechem.OEGetTag("variant"), mol.GetData("variant")
        )  # Copy over variant
        oechem.OESetSDData(
            dockedMol, "name", f"{mol.GetData('name')}-{mol.GetData('variant')}"
        )  # Copy name to SD tag
        oedocking.OESetSDScore(dockedMol, oedock, "docking_score")  # Add to SD tag
        # Sort confs
        oechem.OESortConfsBySDTag(dockedMol, "docking_score", biggerIsBetter=False)
        return dockedMol

    def run_dock(self):
        """
        Run omega in parallel if cluster specified
        """
        # Initialize subprocess
        logger.debug("OEDock called")
        pdock = partial(
            self.dock,
            receptor=self.receptor,
            dock_scoring_function=self.dock_scoring_function,
            dock_search_resolution=self.dock_search_resolution,
            num_poses=self.dock_num_poses,
        )

        if self.client is not None:
            futures = self.client.map(pdock, self.mols)
            self.mols = self.client.gather(futures)
        else:
            self.mols = [pdock(m) for m in self.mols]
        logger.debug("OEDock finished")
        return

    def get_docking_scores(self, smiles: list, return_best_variant: bool = False):
        """
        Read output
        :param smiles: List of SMILES strings
        :param return_best_variant:
        :return optional, list of filenames with best variant
        """
        # Read in docked file
        results = {
            name: dict(
                smiles=smi, **{f"{self.prefix}_{m}": 0.0 for m in self.return_metrics}
            )
            for smi, name in zip(smiles, self.file_names)
        }
        best_score = {name: None for name in self.file_names}
        best_variants = {name: None for name in self.file_names}
        best_mol = {name: None for name in self.file_names}

        # Process mols
        for i, mol in enumerate(self.mols):
            name = mol.GetData("name")
            variant = mol.GetData("variant")
            dscore = list(mol.GetConfs())[0].GetEnergy()  # confs pre-ordered

            if best_score[name] is None:
                best_score[name] = dscore
                best_variants[name] = f"{name}-{variant}"
                best_mol[name] = i
                results[name].update(
                    {
                        f"{self.prefix}_docking_score": dscore,
                        f"{self.prefix}_best_variant": f"{name}-{variant}",
                    }
                )
                # Add charge info
                mol_smiles = oechem.OEMolToSmiles(mol)
                net_charge, positive_charge, negative_charge = (
                    MolecularDescriptors.charge_counts(mol_smiles)
                )
                results[name].update(
                    {
                        f"{self.prefix}_NetCharge": net_charge,
                        f"{self.prefix}_PositiveCharge": positive_charge,
                        f"{self.prefix}_NegativeCharge": negative_charge,
                    }
                )
                logger.debug(f"Docking score for {name}-{variant}: {dscore}")

            elif dscore < best_score[name]:
                best_score[name] = dscore
                best_variants[name] = f"{name}-{variant}"
                best_mol[name] = i
                results[name].update(
                    {
                        f"{self.prefix}_docking_score": dscore,
                        f"{self.prefix}_best_variant": f"{name}-{variant}",
                    }
                )
                # Add charge info
                mol_smiles = oechem.OEMolToSmiles(mol)
                net_charge, positive_charge, negative_charge = (
                    MolecularDescriptors.charge_counts(mol_smiles)
                )
                results[name].update(
                    {
                        f"{self.prefix}_NetCharge": net_charge,
                        f"{self.prefix}_PositiveCharge": positive_charge,
                        f"{self.prefix}_NegativeCharge": negative_charge,
                    }
                )
                logger.debug(f"Found better {name}-{variant}: {dscore}")

            # Otherwise ignore
            else:
                pass

        # Convert results into correctly ordered format and save files
        for smi, name in zip(smiles, self.file_names):
            self.docking_results.append(results[name])
            # Save file
            mol = self.mols[best_mol[name]]
            best_variant = best_variants[name]
            ofs = oechem.oemolostream(
                os.path.join(self.directory, f"{best_variant}_docked.sdf")
            )
            oechem.OEWriteMolecule(ofs, mol)
            ofs.close()

        logger.debug(f"Best scores: {best_score}")
        if return_best_variant:
            best_variants = list(best_variants.values())
            logger.debug(f"Returning best variants: {best_variants}")
            return best_variants

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.client is None):
            parallel = False

        keep_poses = [f"{k}_docked.sdf" for k in keep]
        logger.debug(f"Keeping pose files: {keep_poses}")
        del_files = []
        for name in self.file_names:
            # Grab files
            files = glob.glob(os.path.join(self.directory, f"{name}*"))
            logger.debug(f"Glob found {len(files)} files")

            if len(files) > 0:
                try:
                    files = [
                        file
                        for file in files
                        if "log.txt" not in file
                        and not any([p in file for p in keep_poses])
                    ]

                    if parallel:
                        [del_files.append(file) for file in files]
                    else:
                        [os.remove(file) for file in files]
                # No need to stop if files can't be found and deleted
                except FileNotFoundError:
                    logger.debug("File not found.")
                    pass

        if parallel:
            futures = self.client.map(os.remove, del_files)
            _ = self.client.gather(futures)
        return self

    def score(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores for OEDock given a list of SMILES strings.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step

        # Create log directory
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_OEDock", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_names = file_names
        self.docking_results = []  # make sure no carry over

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

        # Run protocol
        self.variants, variant_files = self.ligand_protocol(
            smiles=smiles, directory=self.directory, file_names=file_names
        )
        self.mols = self.load_prepared_ligand()
        self.run_omega()
        self.run_dock()
        best_variants = self.get_docking_scores(smiles=smiles, return_best_variant=True)

        # Cleanup
        self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None
        self.mols = None

        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results

    def __call__(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores for OEDock given a list of SMILES strings.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles, directory=directory, file_names=file_names)
