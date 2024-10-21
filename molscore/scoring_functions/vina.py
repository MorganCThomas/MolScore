import atexit
import glob
import logging
import os
import re
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
from rdkit import Chem

from molscore import resources
from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import (
    DaskUtils,
    check_openbabel,
    read_mol,
    timedSubprocess,
    check_env,
)

logger = logging.getLogger("vina")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class VinaDock:
    """
    Score structures based on their Smina docking score, using Gypsum-DL for ligand preparation
    """

    return_metrics = [
        "docking_score",
        "inter_score",
        "intra_score",
        "NetCharge",
        "PositiveCharge",
        "NegativeCharge",
        "best_variant",
    ]
    presets = {
        "5HT2A": {
            "receptor": resources.files("molscore.data.structures.6A93").joinpath(
                "6A93p_rec.pdb"
            ),
            "ref_ligand": resources.files("molscore.data.structures.6A93").joinpath(
                "6A93p_risperidone.sdf"
            ),
        },
        "5HT2A-3x32": {
            "receptor": resources.files("molscore.data.structures.6A93").joinpath(
                "6A93p_rec.pdb"
            ),
            "ref_ligand": resources.files("molscore.data.structures.6A93").joinpath(
                "6A93p_risperidone.sdf"
            ),
            "dock_constraints": resources.files(
                "molscore.data.structures.6A93"
            ).joinpath("6A93p_3x32Cat.restr"),
        },
        "DRD2": {
            "receptor": resources.files("molscore.data.structures.6CM4").joinpath(
                "6CM4p_rec.pdb"
            ),
            "ref_ligand": resources.files("molscore.data.structures.6CM4").joinpath(
                "6CM4p_risperidone.sdf"
            ),
        },
        "BACE1_4B05": {
            "receptor": resources.files("molscore.data.structures.4B05").joinpath(
                "4B05p_rec.pdbqt"
            ),
            "ref_ligand": resources.files("molscore.data.structures.4B05").joinpath(
                "AZD3839_rdkit.mol"
            ),
        },
    }

    @staticmethod
    def check_installation():
        if not check_env("vina"):
            raise RuntimeError(
                "Vina not found in PATH, please export the path to the vina executable as 'vina'"
                )

    def __init__(
        self,
        prefix: str,
        preset: str = None,
        receptor: Union[str, os.PathLike] = None,
        ref_ligand: Union[str, os.PathLike] = None,
        cpus: int = 1,
        cluster: Union[str, int] = None,
        file_preparation: str = "obabel",
        ligand_preparation: str = "GypsumDL",
        prep_timeout: float = 60.0,
        dock_scoring: str = "vina",
        dock_timeout: float = 120.0,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param preset: Pre-populate file paths for receptors, reference ligand and/or constraints etc. [5HT2A, 5HT2A-3x32, DRD2, BACE1_4B05]
        :param receptor: Path to receptor file (.pdb, .pdbqt)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf, .pdb, .mol2)
        :param cpus: Number of Vina CPUs to use per simulation
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param file_preparation: Which software to use to prepare files for docking [obabel, mgltools]
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_scoring: Docking scoring function to use [vina, vinardo]
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation
        """
        self.subprocess = timedSubprocess()

        # Check requirements
        check_openbabel()
        self.check_installation()

        # Set file prep environment
        self.prep_env = file_preparation
        if self.prep_env == "mgltools":
            assert (
                "mgltools" in os.environ.keys()
            ), "Can't find mgltools in PATH, please export the path the mgtltools 1.5.6 as 'mgltools'"
            self.prep_env = os.path.join(os.environ["mgltools"], "bin", "pythonsh")
            self.prep_rec = os.path.join(
                os.environ["mgltools"],
                "MGLToolsPckgs/AutoDockTools/Utilities24",
                "prepare_receptor4.py",
            )
            self.prep_lig = os.path.join(
                os.environ["mgltools"],
                "MGLToolsPckgs/AutoDockTools/Utilities24",
                "prepare_ligand4.py",
            )

        # Check if preset is provided
        assert preset or (
            receptor and ref_ligand
        ), "Either preset or receptor and ref_ligand must be provided"
        if preset:
            assert (
                preset in self.presets.keys()
            ), f"preset must be one of {self.presets.keys()}"
            receptor = str(self.presets[preset]["receptor"])
            ref_ligand = str(self.presets[preset]["ref_ligand"])

        # If receptor is pdb, convert
        if not receptor.endswith(".pdbqt"):
            pdbqt_receptor = receptor.rsplit(".", 1)[0] + ".pdbqt"
            if self.prep_env == "obabel":
                out, err = self.subprocess.run(
                    " ".join([self.prep_env, receptor, "-O", pdbqt_receptor])
                )
            else:
                out, err = self.subprocess.run(
                    " ".join(
                        [
                            self.prep_env,
                            self.prep_rec,
                            "-r",
                            receptor,
                            "-o",
                            pdbqt_receptor,
                            "-A",
                            "checkhydrogens",
                        ]
                    )
                )
            receptor = pdbqt_receptor

        # Find get centroid of ref ligand
        lig_mol = read_mol(ref_ligand)
        if lig_mol:
            lig_mol = Chem.RemoveHs(lig_mol)
            c = lig_mol.GetConformer()
            self.ref_xyz = np.mean(c.GetPositions(), axis=0)
        else:
            raise ValueError(
                "Error parsing reference ligand, cannot calculate docking box"
            )

        # If ligand is not pdbqt, convert
        if not ref_ligand.endswith(".pdbqt"):
            pdbqt_ligand = ref_ligand.rsplit(".", 1)[0] + ".pdbqt"
            if self.prep_env == "obabel":
                out, err = self.subprocess.run(
                    " ".join([self.prep_env, ref_ligand, "-O", pdbqt_ligand])
                )
            else:
                # Prepare_ligand4 only takes pdb or mol2, so convert first
                if not ref_ligand.endswith(".pdb"):
                    out, err = self.subprocess.run(
                        " ".join(
                            [
                                "obabel",
                                ref_ligand,
                                "-O",
                                ref_ligand.rsplit(".", 1)[0] + ".pdb",
                            ]
                        )
                    )
                    ref_ligand = ref_ligand.rsplit(".", 1)[0] + ".pdb"
                out, err = self.subprocess.run(
                    " ".join(
                        [
                            self.prep_env,
                            self.prep_lig,
                            "-l",
                            ref_ligand,
                            "-o",
                            pdbqt_ligand,
                        ]
                    )
                )
            ref_ligand = pdbqt_ligand

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.receptor = os.path.abspath(receptor)
        self.ref = os.path.abspath(ref_ligand)
        self.file_names = None
        self.variants = None
        self.cpus = cpus
        self.vina_env = os.environ["vina"]
        self.dock_scoring = dock_scoring  # ad4 requires flex receptor and grid / map
        self.dock_timeout = float(dock_timeout)
        self.prep_timeout = float(prep_timeout)
        self.temp_dir = TemporaryDirectory()

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

    def _close_dask(self):
        if self.client:
            self.client.close()
            self.client.cluster.close()

    def ligands2pdbqt(self, variants, variant_files):
        """Reformat prepared ligands to .pdbqt"""
        cmds1 = []
        cmds2 = []
        new_variant_files = []
        p = timedSubprocess(timeout=self.prep_timeout).run
        logger.debug("Converting ligands to pdbqt")
        for vfile in variant_files:
            new_vfile = vfile.replace(".sdf", ".pdbqt")
            new_variant_files.append(new_vfile)
            if self.prep_env == "obabel":
                cmds2.append(f"{self.prep_env} {vfile} -O {new_vfile}")
            else:
                # Convert to pdb first
                tvfile = vfile.replace(".sdf", ".pdb")
                cmds1.append(f"obabel {vfile} -O {tvfile}")
                cmds2.append(
                    f"{self.prep_env} {self.prep_lig} -l {tvfile} -o {new_vfile}"
                )

        if self.client:
            if cmds1:
                futures = self.client.map(p, cmds1)
                self.client.gather(futures)
            futures = self.client.map(p, cmds2)
            self.client.gather(futures)
        else:
            if cmds1:
                _ = [p(cmd) for cmd in cmds1]
            _ = [p(cmd) for cmd in cmds2]
        logger.debug("Ligands converted to pdbqt")
        return variants, new_variant_files

    def ligands2sdf(self):
        """Look for docked files and convert to sdf"""
        p = timedSubprocess(timeout=self.prep_timeout).run
        futures = []
        for docked_file in glob.glob(os.path.join(self.directory, "*_docked.pdbqt")):
            new_docked_file = docked_file.replace(".pdbqt", ".sdf")
            cmd = f"obabel {docked_file} -O {new_docked_file}"
            if self.client:
                futures.append(self.client.submit(p, cmd))
            else:
                p(cmd)
        if self.client:
            self.client.gather(futures)

    def dock_ligands(self, variant_paths):
        vina_commands = []
        log_paths = []
        for lpath in variant_paths:
            out_file = os.path.join(
                self.directory,
                os.path.basename(lpath).replace("_prepared.pdbqt", "_docked.pdbqt"),
            )
            cmd = (
                f"{self.vina_env} --receptor {self.receptor} --ligand {lpath} --out {out_file} "
                f"--center_x {self.ref_xyz[0]:.2f} --center_y {self.ref_xyz[1]:.2f} --center_z {self.ref_xyz[2]:.2f} "
                f"--size_x 10.0 --size_y 10.0 --size_z 10.0 "
                f"--scoring {self.dock_scoring} --cpu {self.cpus} --exhaustiveness 8"
            )
            vina_commands.append(cmd)

        # Initialize subprocess
        logger.debug("Vina called")
        p = timedSubprocess(timeout=self.dock_timeout).run

        if self.client is not None:
            futures = self.client.map(p, vina_commands)
            results = self.client.gather(futures)
        else:
            results = [p(command) for command in vina_commands]
        logger.debug("Vina finished")
        _ = [logger.warning(err.decode()) for out, err in results if err != "".encode()]
        return log_paths

    @staticmethod
    def parse_vina_score(vina_text: str):
        """Parse REMARK in SDF files leftover"""
        m = re.search("^ VINA RESULT:(.*)\n", vina_text)
        score = float(m.groups()[0].split()[0])

        m = re.search("\n INTER:(.*)\n", vina_text)
        inter = float(m.groups()[0].strip())

        m = re.search("\n INTRA:(.*)\n", vina_text)
        intra = float(m.groups()[0].strip())
        return score, inter, intra

    def get_docking_scores(self, smiles: list, return_best_variant: bool = False):
        """
        Read output sdfs, get output properties
        :param smiles: List of SMILES strings
        :param return_best_variant:
        :return optional, list of filenames with best variant
        """
        # Read in docked file
        best_variants = self.file_names.copy()
        best_score = {name: None for name in self.file_names}

        # For each molecule
        for i, (smi, name) in enumerate(zip(smiles, self.file_names)):
            docking_result = {"smiles": smi}

            # If no variants enumerate 0
            if len(self.variants[name]) == 0:
                logger.debug(f"{name}_docked.sd does not exist")
                if best_score[name] is None:  # Only if no other score for prefix
                    docking_result.update(
                        {f"{self.prefix}_" + k: 0.0 for k in self.return_metrics}
                    )
                    logger.debug("Returning 0.0 as no variants exist")

            # For each variant
            for variant in self.variants[name]:
                # Get associated Mol
                out_file = os.path.join(self.directory, f"{name}-{variant}_docked.sdf")
                if os.path.exists(out_file):
                    try:
                        vina_out = Chem.ForwardSDMolSupplier(out_file, sanitize=False)
                        mol = next(vina_out)  # Mol file is ordered by dscore
                        mol = manually_charge_mol(
                            Chem.AddHs(mol)
                        )  # TODO incorrectly computing charge, RDKit bug doesn't add Hs
                        dscore, interscore, intrascore = self.parse_vina_score(
                            mol.GetPropsAsDict()["REMARK"]
                        )

                        # If molecule doesn't have a score yet append it and the variant
                        if best_score[name] is None:
                            best_score[name] = dscore
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_docking_score": dscore,
                                    f"{self.prefix}_inter_score": interscore,
                                    f"{self.prefix}_intra_score": intrascore,
                                }
                            )
                            # Add charge info
                            net_charge, positive_charge, negative_charge = (
                                MolecularDescriptors.charge_counts(mol)
                            )
                            docking_result.update(
                                {
                                    f"{self.prefix}_NetCharge": net_charge,
                                    f"{self.prefix}_PositiveCharge": positive_charge,
                                    f"{self.prefix}_NegativeCharge": negative_charge,
                                }
                            )
                            logger.debug(
                                f"Docking score for {name}-{variant}: {dscore}"
                            )
                        # If docking score is better change it...
                        elif dscore < best_score[name]:
                            best_score[name] = dscore
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_docking_score": dscore,
                                    f"{self.prefix}_inter_score": interscore,
                                    f"{self.prefix}_intra_score": intrascore,
                                }
                            )
                            # Add charge info
                            net_charge, positive_charge, negative_charge = (
                                MolecularDescriptors.charge_counts(mol)
                            )
                            docking_result.update(
                                {
                                    f"{self.prefix}_NetCharge": net_charge,
                                    f"{self.prefix}_PositiveCharge": positive_charge,
                                    f"{self.prefix}_NegativeCharge": negative_charge,
                                }
                            )
                            logger.debug(f"Found better {name}-{variant}: {dscore}")

                    # If parsing the molecule threw an error and nothing stored, append 0
                    except Exception:
                        logger.debug(f"Error processing {name}-{variant}_docked.sdf")
                        if (
                            best_score[name] is None
                        ):  # Only if no other score for prefix
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_" + k: 0.0
                                    for k in self.return_metrics
                                }
                            )
                            logger.debug(
                                "Returning 0.0 unless a successful variant is found"
                            )

                else:
                    logger.debug(f"{name}-{variant}_docked.sdf does not exist")
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f"{name}-{variant}"
                        docking_result.update(
                            {f"{self.prefix}_" + k: 0.0 for k in self.return_metrics}
                        )
                        logger.debug(
                            "Returning 0.0 unless a successful variant is found"
                        )

            # Add best variant information to docking result
            docking_result.update({f"{self.prefix}_best_variant": best_variants[i]})
            self.docking_results.append(docking_result)

        logger.debug(f"Best scores: {best_score}")
        if return_best_variant:
            logger.debug(f"Returning best variants: {best_variants}")
            return best_variants

        return self

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

    def __call__(
        self,
        smiles: list,
        directory: str,
        file_names: list,
        cleanup: bool = True,
        **kwargs,
    ):
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        self.file_names = file_names
        self.docking_results = []

        # Create log directory
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_VinaDock", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f"{step}_log.txt"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Prepare ligands
        self.variants, variant_paths = self.ligand_protocol(
            smiles=smiles, directory=self.directory, file_names=self.file_names
        )
        self.variants, variant_paths = self.ligands2pdbqt(self.variants, variant_paths)

        # Dock ligands
        self.dock_ligands(variant_paths)
        self.ligands2sdf()

        # Process output
        best_variants = self.get_docking_scores(smiles, return_best_variant=True)

        # Cleanup
        if cleanup:
            self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None

        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results


def manually_charge_mol(mol):
    # TODO vina will only populate polar hydrogens, RDKit bug won't add Hs to file.
    PT = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        v = atom.GetExplicitValence()
        dv = PT.GetDefaultValence(atom.GetAtomicNum())
        if (v < dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(-1)
        if (v > dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return Chem.RemoveHs(mol)
