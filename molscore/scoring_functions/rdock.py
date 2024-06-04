import atexit
import glob
import logging
import os
import shutil
from functools import partial
from tempfile import TemporaryDirectory
from typing import Union

from rdkit.Chem import AllChem as Chem

from molscore import resources
from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import (
    DaskUtils,
    check_openbabel,
    timedFunc2,
    timedSubprocess,
)

logger = logging.getLogger("rdock")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class rDock:
    """
    Scores structures based on their rDock docking score
    """

    return_metrics = [
        "SCORE",
        "SCORE.INTER",
        "SCORE.INTRA",
        "TETHERED.RMSD",
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
        if shutil.which("rbdock") is None:
            raise RuntimeError(
                "Could not find rDock path, please ensure proper installation."
            )

    @staticmethod
    def cavity_config(receptor_file, ligand_file=None, xyz=None, cavity_kwargs={}):
        config = f"""RBT_PARAMETER_FILE_V1.00
TITLE gart_DUD

RECEPTOR_FILE {os.path.basename(receptor_file)}
RECEPTOR_FLEX 3.0
"""
        assert ligand_file or xyz, "Either ligand_file or xyz must be provided"
        if ligand_file:
            config += f"""
##################################################################
### CAVITY DEFINITION: REFERENCE LIGAND METHOD
##################################################################
SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL {os.path.basename(ligand_file)}
    RADIUS {cavity_kwargs['RADIUS']}
    SMALL_SPHERE {cavity_kwargs['SMALL_SPHERE']}
    MIN_VOLUME {cavity_kwargs['MIN_VOLUME']}
    MAX_CAVITIES {cavity_kwargs['MAX_CAVITIES']}
    VOL_INCR {cavity_kwargs['VOL_INCR']}
GRIDSTEP {cavity_kwargs['GRIDSTEP']}
END_SECTION
"""
        else:
            config += f"""
##################################################################
### CAVITY DEFINITION: SPHERE METHOD
##################################################################
SECTION MAPPER
    SITE_MAPPER RbtSphereSiteMapper
    CENTER ({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})
    RADIUS {cavity_kwargs['RADIUS']}
    SMALL_SPHERE {cavity_kwargs['SMALL_SPHERE']}
    LARGE_SPHERE {cavity_kwargs['LARGE_SPHERE']}
    MAX_CAVITIES {cavity_kwargs['MAX_CAVITIES']}
GRIDSTEP {cavity_kwargs['GRIDSTEP']}
END_SECTION
"""

        config += """
#################################
#CAVITY RESTRAINT PENALTY
#################################
SECTION CAVITY
    SCORING_FUNCTION RbtCavityGridSF
    WEIGHT 1.0
END_SECTION
"""
        return config

    @staticmethod
    def add_pharmacophore_constraint(
        config, constraints=None, optional_constraints=None, n_optional=1
    ):
        if optional_constraints:
            config += f"""
#################################
## PHARMACOPHORIC RESTRAINTS
#################################
SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    CONSTRAINTS_FILE {os.path.basename(constraints)}
    OPTIONAL_FILE {os.path.basename(optional_constraints)}
    NOPT {n_optional}
END_SECTION
"""
        else:
            config += f"""
#################################
## PHARMACOPHORIC RESTRAINTS
#################################
SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    CONSTRAINTS_FILE {os.path.basename(constraints)}
END_SECTION
"""
        return config

    @staticmethod
    def add_tether_constraint(
        config,
        translation="TETHERED",
        rotation="TETHERED",
        dihedral="FREE",
        max_trans=1.0,
        max_rot=30.0,
    ):
        assert all(
            param in ["FIXED", "TETHERED", "FREE"]
            for param in [translation, rotation, dihedral]
        ), "translation, rotation and dihedral must be one of FIXED, TETHERED or FREE"
        config += f"""
#################################
## LIGAND RESTRAINTS
#################################
SECTION LIGAND
   TRANS_MODE {translation}
   ROT_MODE {rotation}
   DIHEDRAL_MODE {dihedral}
   MAX_TRANS {max_trans:.1f}
   MAX_ROT {max_rot:.1f}
END_SECTION
"""
        return config

    def __init__(
        self,
        prefix: str,
        preset: str = None,
        receptor: Union[str, os.PathLike] = None,
        ref_ligand: Union[str, os.PathLike] = None,
        ref_xyz: list = None,
        cavity_kwargs: dict = {
            "RADIUS": 10.0,
            "SMALL_SPHERE": 2.0,
            "LARGE_SPHERE": 5.0,
            "MAX_CAVITIES": 1,
            "MIN_VOLUME": 100,
            "VOL_INCR": 0.0,
            "GRIDSTEP": 0.5,
        },
        cluster: Union[str, int] = None,
        ligand_preparation: str = "GypsumDL",
        ligand_preparation_kwargs: dict = {},
        prep_timeout: float = 30.0,
        dock_protocol: Union[str, os.PathLike] = "dock",
        dock_timeout: float = 120.0,
        n_runs: int = 5,
        dock_substructure_constraints: str = None,
        dock_substructure_max_trans: float = 0.5,
        dock_substructure_max_rot: int = 10.0,
        dock_substructure_max_rmsd: float = 2.0,
        dock_constraints: Union[str, os.PathLike] = None,
        dock_opt_constraints: Union[str, os.PathLike] = None,
        dock_n_opt_constraints: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param preset: Pre-populate file paths for receptors, reference ligand and/or constraints etc. [5HT2A, 5HT2A-3x32, DRD2, BACE1_4B05]
        :param receptor: Path to receptor file (.pdb)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf, .pdb)
        :param ref_xyz: XYZ coordinates of the centre of the docking box
        :param cavity_kwargs: Dictionary of parameters for cavity generation, any changes will update defaults so only need to supply the changing parameter, see rDock manual for details
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param ligand_preparation_kwargs: Additional keyword arguments for ligand preparation
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_protocol: Select from docking protocols or path to a custom .prm protocol [dock, dock_solv, dock_grid, dock_solv_grid, minimise, minimise_solv, score, score_solv]
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation
        :param n_runs: Number of docking trials (poses to return)
        :param dock_substructure_constraints: SMARTS pattern to constrain ligands to reference ligand
        :param dock_substructure_max_trans: Maximum allowed translation of tethered substructure
        :param dock_substructure_max_rot: Maximum allowed rotation of tethered substructure
        :param dock_substructure_max_rmsd: Maximum allowed RMSD of tethered substructure
        :param dock_constraints: Path to rDock pharmacophoric constriants file that are mandatory
        :param dock_opt_constraints: Path to rDock pharmacophoric constriants file that are optional
        :param dock_n_opt_constraints: Number of optional constraints required
        """
        # Check rDock installation
        self.check_installation()
        check_openbabel()

        # Check if preset is provided
        assert preset or (
            receptor and (ref_ligand or ref_xyz)
        ), "Either preset or receptor and ref_ligand/ref_xyz must be provided"
        if preset:
            assert (
                preset in self.presets.keys()
            ), f"preset must be one of {self.presets.keys()}"
            receptor = str(self.presets[preset]["receptor"])
            ref_ligand = str(self.presets[preset]["ref_ligand"])
            if "dock_constraints" in self.presets[preset].keys():
                dock_constraints = str(self.presets[preset]["dock_constraints"])
            if "dock_opt_constraints" in self.presets[preset].keys():
                dock_opt_constraints = str(self.presets[preset]["dock_opt_constraints"])

        # Control subprocess in/out
        self.subprocess = timedSubprocess(timeout=None, shell=True)

        # If receptor is pdb, convert
        assert os.path.exists(receptor), f"Receptor file {receptor} not found"
        if not receptor.endswith(".mol2"):
            mol2_receptor = receptor.rsplit(".", 1)[0] + ".mol2"
            self.subprocess.run(f"obabel {receptor} -O {mol2_receptor}")
            receptor = os.path.abspath(mol2_receptor)

        # If ref_ligand doesn't end with sd, replace
        if ref_ligand:
            assert os.path.exists(
                ref_ligand
            ), f"Reference ligand file {ref_ligand} not found"
            if not ref_ligand.endswith(".sd"):
                sd_ligand = ref_ligand.rsplit(".", 1)[0] + ".sd"
                self.subprocess.run(f"obabel {ref_ligand} -O {sd_ligand}")
                ref_ligand = os.path.abspath(sd_ligand)

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.receptor = receptor
        self.ref = ref_ligand
        self.xyz = ref_xyz
        self.cavity_kwargs = {
            "RADIUS": 10.0,
            "SMALL_SPHERE": 2.0,
            "LARGE_SPHERE": 5.0,
            "MAX_CAVITIES": 1,
            "MIN_VOLUME": 100,
            "VOL_INCR": 0.0,
            "GRIDSTEP": 0.5,
        }
        self.cavity_kwargs.update(cavity_kwargs)
        self.file_names = None
        self.variants = None
        self.dock_timeout = float(dock_timeout)
        if "timeout" in kwargs.items():
            self.dock_timeout = float(kwargs["timeout"])  # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.temp_dir = TemporaryDirectory()
        self.n_runs = n_runs
        self.rdock_env = "rbdock"
        self.rcav_env = "rbcavity"
        self.substructure_smarts = dock_substructure_constraints
        self.substructure_max_rmsd = dock_substructure_max_rmsd
        self.dock_constraints = (
            os.path.abspath(dock_constraints) if dock_constraints else dock_constraints
        )
        self.dock_opt_constraints = (
            os.path.abspath(dock_opt_constraints)
            if dock_opt_constraints
            else dock_opt_constraints
        )
        self.dock_n_opt_constraints = dock_n_opt_constraints
        self.rdock_files = [self.receptor]
        if self.ref:
            self.rdock_files.append(self.ref)

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
                dask_client=self.client,
                timeout=self.prep_timeout,
                logger=logger,
                **ligand_preparation_kwargs,
            )
        else:
            self.ligand_protocol = self.ligand_protocol(
                logger=logger, **ligand_preparation_kwargs
            )

        # Select docking protocol
        if dock_protocol in [
            "dock",
            "dock_solv",
            "dock_grid",
            "dock_solv_grid",
            "minimise",
            "minimise_solv",
            "score",
            "score_solv",
        ]:
            self.dock_protocol = dock_protocol + ".prm"
        else:
            self.dock_protocol = os.path.abspath(dock_protocol)

        # Prepare grid file in tempfiles
        os.environ["RBT_HOME"] = self.temp_dir.name
        # Write config
        rec_prm = self.cavity_config(
            self.receptor, self.ref, self.xyz, self.cavity_kwargs
        )
        # Add PH4 constriants
        if self.dock_constraints or self.dock_opt_constraints:
            # If only optional provided we have to create an empty mandatory file
            if self.dock_opt_constraints and not self.dock_constraints:
                self.dock_constraints = os.path.join(self.temp_dir.name, "empty.restr")
                with open(self.dock_constraints, "wt") as f:
                    pass
            rec_prm = self.add_pharmacophore_constraint(
                rec_prm,
                self.dock_constraints,
                self.dock_opt_constraints,
                self.dock_n_opt_constraints,
            )
            if self.dock_constraints:
                self.rdock_files.append(self.dock_constraints)
            if self.dock_opt_constraints:
                self.rdock_files.append(self.dock_opt_constraints)
        # Add substructure constraints
        if self.substructure_smarts:
            rec_prm = self.add_tether_constraint(
                rec_prm,
                max_trans=dock_substructure_max_trans,
                max_rot=dock_substructure_max_rot,
            )
            with Chem.SDMolSupplier(self.ref, removeHs=False) as suppl:
                self.ref_mol = suppl[0]
            assert self.ref_mol.GetSubstructMatch(
                Chem.MolFromSmarts(self.substructure_smarts)
            ), f"Substructure {self.substructure_smarts} not found in reference ligand"
        # Copy files to RBT home
        self.subprocess.run(f'cp {" ".join(self.rdock_files)} {self.temp_dir.name}')
        # Write config
        self.rec_prm = os.path.join(self.temp_dir.name, "cavity.prm")
        with open(self.rec_prm, "wt") as f:
            f.write(rec_prm)
        self.rdock_files.append(self.rec_prm)
        # Prepare cavity
        self.subprocess.run(f"{self.rcav_env} -was -d -r {self.rec_prm}")
        self.cavity = os.path.join(self.temp_dir.name, "cavity.as")
        grid = os.path.join(self.temp_dir.name, "cavity_cav1.grd")
        self.rdock_files.extend([self.cavity, grid])

    def _move_rdock_files(self, cwd):
        os.environ["RBT_HOME"] = cwd
        self.subprocess.run(f'cp {" ".join(self.rdock_files)} {cwd}')

    @staticmethod
    def _align_mol(
        query: os.PathLike,
        ref_mol: Chem.rdchem.Mol,
        smarts: str,
        max_rmsd: float,
        logger: logging.Logger,
    ):
        # NOTE could return several variants for multiple ref matches? E.g., variant-1_205 i.e.,  +0{count}
        # Load query file
        with Chem.SDMolSupplier(query, removeHs=False) as suppl:
            query_mol = suppl[0]
        # Load SMARTS and get atom matches
        patt = Chem.MolFromSmarts(smarts)
        query_match = query_mol.GetSubstructMatch(patt)
        ref_match = ref_mol.GetSubstructMatch(patt)
        if query_match:
            # Do a conf search and align each conf
            Chem.EmbedMultipleConfs(query_mol, numConfs=20, ETversion=2)
            rmsds = []
            for i in range(query_mol.GetNumConformers()):
                # Minimize conf
                try:
                    ff = Chem.UFFGetMoleculeForceField(query_mol, confId=i)
                    ff.Minimize()
                except Exception:
                    pass
                # Align to scaff
                rmsds.append(
                    Chem.AlignMol(
                        query_mol,
                        ref_mol,
                        prbCid=i,
                        atomMap=list(zip(query_match, ref_match)),
                    )
                )
            # Select lowest RMSD
            query_mol = Chem.Mol(
                query_mol, confId=sorted(enumerate(rmsds), key=lambda x: x[1])[0][0]
            )
            # Set tethered atoms (rDock indexing starts from 1)...
            query_match = list(query_match)
            query_mol.SetProp(
                "TETHERED ATOMS", ",".join([str(a + 1) for a in query_match])
            )
            query_mol.SetProp("TETHERED.RMSD", str(min(rmsds)))
            # Save aligned mol if below max_rmsd
            if max_rmsd:
                if min(rmsds) <= max_rmsd:
                    with Chem.SDWriter(query) as writer:
                        writer.write(query_mol)
                else:
                    # Delete file so that it can't be run
                    os.remove(query)
            else:
                with Chem.SDWriter(query) as writer:
                    writer.write(query_mol)
        else:
            # TODO unspecified constrained docking by MCS
            logger.warning(
                f"No substructure match found for {query}, skipping."
            )
            os.remove(query)

    def align_mols(self, varients, varient_files):
        logger.debug("Aligning molecules for tethered docking")
        if self.client:
            p = timedFunc2(self._align_mol, timeout=self.prep_timeout)
            p = partial(
                p,
                ref_mol=self.ref_mol,
                smarts=self.substructure_smarts,
                max_rmsd=self.substructure_max_rmsd,
                logger=logger,
            )
            futures = self.client.map(p, varient_files)
            _ = self.client.gather(futures)
        else:
            for vfile in varient_files:
                p = timedFunc2(self._align_mol, timeout=self.prep_timeout)
                p(
                    vfile,
                    ref_mol=self.ref_mol,
                    smarts=self.substructure_smarts,
                    max_rmsd=self.substructure_max_rmsd,
                    logger=logger,
                )
        return varients, varient_files

    @staticmethod
    def _fixup_ph4_sdf(sdf_file):
        """When using ph4 constraints, rDock doesn't write SDF files properly and misses the final '>' in properties"""
        with open(sdf_file, 'rt') as f:
            fixed_lines = []
            for line in f:
                if line.startswith(">") and not line.endswith(">\n"):
                    line = line[:-1] + ">\n"
                fixed_lines.append(line)
        with open(sdf_file, 'wt') as f:
            f.writelines(fixed_lines)

    def reformat_ligands(self, varients, varient_files):
        """Reformat prepared ligands to .sd via obabel (RDKit doesn't write charge in a rDock friendly way)"""
        futures = []
        new_varient_files = []
        for vfile in varient_files:
            new_vfile = vfile.replace(".sdf", ".sd")
            new_varient_files.append(new_vfile)
            if self.client:
                p = timedSubprocess()
                futures.append(
                    self.client.submit(p.run, f"obabel {vfile} -O {new_vfile}")
                )
            else:
                self.subprocess.run(f"obabel {vfile} -O {new_vfile}")

        # Wait for parallel jobs
        if self.client:
            self.client.gather(futures)

        return varients, new_varient_files

    def run_rDock(self, ligand_paths):
        # Move input files and set env
        self._move_rdock_files(self.directory)
        # Prepare rDock commands
        rdock_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                in_lig = os.path.join(self.directory, f"{name}-{variant}_prepared.sd")
                out_lig = os.path.join(self.directory, f"{name}-{variant}_docked")
                command = f"{self.rdock_env} -i {in_lig} -o {out_lig} -r cavity.prm -p {self.dock_protocol} -n {self.n_runs} -allH"
                rdock_commands.append(command)

        # Initialize subprocess
        logger.debug("rDock called")
        p = timedSubprocess(timeout=self.dock_timeout)
        p = partial(p.run, cwd=self.directory)

        # Submit docking subprocesses
        if self.client is not None:
            futures = self.client.map(p, rdock_commands)
            _ = self.client.gather(futures)
        else:
            _ = [p(command) for command in rdock_commands]
        logger.debug("rDock finished")
        return self

    def get_docking_scores(self, smiles, return_best_variant=True):
        # Iterate over variants
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
                out_file = os.path.join(self.directory, f"{name}-{variant}_docked.sd")
                if os.path.exists(out_file):
                    # Try to load it in, and grab the score
                    try:
                        # NOTE Fixup buggy SDF if using PH4 constraints i.e., rDock sd formatting without ">""
                        if self.dock_constraints or self.dock_opt_constraints:
                            self._fixup_ph4_sdf(out_file)
                        rdock_out = Chem.SDMolSupplier(out_file, sanitize=False)
                        for mol_idx in range(len(rdock_out)):
                            mol = rdock_out[mol_idx]
                            mol = manually_charge_mol(
                                mol
                            )  # RDKit doesn't like rDock charged files
                            dscore = mol.GetPropsAsDict()["SCORE"]

                            # If molecule doesn't have a score yet append it and the variant
                            if best_score[name] is None:
                                best_score[name] = dscore
                                best_variants[i] = f"{name}-{variant}"
                                docking_result.update(
                                    {
                                        f"{self.prefix}_" + k: v
                                        for k, v in mol.GetPropsAsDict().items()
                                        if k in self.return_metrics
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
                                        f"{self.prefix}_" + k: v
                                        for k, v in mol.GetPropsAsDict().items()
                                        if k in self.return_metrics
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

                            # Otherwise ignore
                            else:
                                pass

                    # If parsing the molecule threw an error and nothing stored, append 0
                    except Exception:
                        logger.debug(
                            f"Error processing {name}-{variant}_docked.sd file"
                        )
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

                # If path doesn't exist and nothing store, append 0
                else:
                    logger.debug(f"{name}-{variant}_docked.sd does not exist")
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

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.client is None):
            parallel = False

        keep_poses = [f"{k}_docked.sd" for k in keep]
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
            os.path.abspath(directory), f"{self.prefix}_rDock", step
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
            smiles=smiles,
            directory=self.directory,
            file_names=self.file_names,
            logger=logger,
        )
        if self.substructure_smarts:
            self.variants, variant_paths = self.align_mols(self.variants, variant_paths)
        self.variants, variant_paths = self.reformat_ligands(
            self.variants, variant_paths
        )

        # Dock ligands
        self.run_rDock(variant_paths)

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
    """Manually charge non-full valent atoms, assuming all explicit hydrogens are present"""
    PT = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        e = atom.GetSymbol()
        # Skip sulfur
        if e == "S":
            continue
        v = atom.GetExplicitValence()
        dv = PT.GetDefaultValence(atom.GetAtomicNum())
        if (v < dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(-1)
        if (v > dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return Chem.RemoveHs(mol)
