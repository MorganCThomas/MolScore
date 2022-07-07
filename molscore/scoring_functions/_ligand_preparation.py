import os
import subprocess

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions, GetStereoisomerCount

from molscore.utils.gypsum_dl.Parallelizer import Parallelizer
from molscore.utils.gypsum_dl.Start import prepare_smiles, prepare_3d, add_mol_id_props
from molscore.utils.gypsum_dl.MolContainer import MolContainer

from molscore.scoring_functions.utils import timedSubprocess, get_mol

# Protonation states: LigPrep / Epik / Moka / GypsumDL (Substruct Dist) / OBabel? / OpenEye?
# Tautomers: LigPrep / Epik / Moka / GypsumDL (MolVS)
# Stereoisomers: LigPrep / RDKit / Corina / GypsumDL (manual/rdkit)
# 3D Embedding: LigPrep / GypsumDL(RDKit) / Corina / RDKit / OBabel? / OpenEye?


class LigandPreparation:
    """
    Class to collect ligand preparation protocols required for any Docking scoring functions
    """
    def __init__(self, dask_client = None, timeout: float = 120.0, logger = None):
        self.cluster = False if dask_client is None else dask_client
        self.timeout = timeout
        self.logger = False if logger is None else logger

    def prepare(self, smiles, directory, file_names):
        # This should be overwritten by specific classes
        raise NotImplementedError

    def __call__(self, smiles: list, file_names: list, directory: os.PathLike):
        return self.prepare(smiles=smiles, directory=directory, file_names=file_names)


class LigPrep(LigandPreparation):
    def __init__(self, dask_client=None, timeout: float = 120.0, logger=None, pH: float = 7.0, pHt: float = 1.0, bff: int = 16, max_stereo: int = 8):
        """
        Initialize LigPrep ligand preparation
        :param dask_client: Scheduler address for dask parallelization
        :param timeout: Timeout for subprocess before killing
        :param logger: Currently used logger if present
        :param pH: pH at which to protonate molecules at
        :param pHt: pH Tolerance
        :param bff: Which forcefield to use, 14=OPLS3_2005 or 16=OPLS3e [14, 16]
        :param s: Maximum number of stereoisomers to generate, default is 32 but this is quite inefficient
        """
        super().__init__(dask_client=dask_client, timeout=timeout, logger=logger)
        self.pH = pH
        self.pHt = pHt
        self.bff = bff
        self.max_stereo = max_stereo
        # Setup env
        try:
            self.env = os.path.join(os.environ['SCHRODINGER'], 'ligprep')
        except KeyError:
            raise KeyError("Please ensure you have a Schrodinger installation and license")

    def prepare(self, smiles: list, directory: str, file_names: list):
        """
        Call ligprep to prepare molecules.
        :param smiles: List of SMILES strings
        :param directory: Output directory
        :param file_names: File names for SMILES
        :return: (dictionary of variants, file_paths)
        """
        # Write out smiles to sdf files and prepare ligprep commands
        ligprep_commands = []
        for smi, name in zip(smiles, file_names):
            smi_in = os.path.join(directory, f'{name}.smi')
            sdf_out = os.path.join(directory, f'{name}_prepared.sdf')

            with open(smi_in, 'w') as f:
                f.write(smi)

            command = " ".join((self.env,
                                f"-ismi {smi_in}",
                                f"-osd {sdf_out}",
                                f"-ph {self.pH}",
                                f"-pht {self.pHt}",
                                f"-bff {self.bff}",
                                f"-s {self.max_stereo}",
                                "-epik",
                                "-WAIT",
                                "-NOJOBID"))
            ligprep_commands.append(command)

        # Initialize subprocess
        if self.logger: self.logger.debug('LigPrep called')
        p = timedSubprocess(timeout=self.timeout).run

        # Run commands either using Dask or sequentially
        if self.cluster:
            futures = self.cluster.map(p, ligprep_commands)
            _ = self.cluster.gather(futures)
        else:
            _ = [p(command) for command in ligprep_commands]
        if self.logger: self.logger.debug('LigPrep finished')

        # Read in ligprep output files and split to individual variants
        variants = {name: [] for name in file_names}
        variant_files = []
        for name in file_names:
            out_file = os.path.join(directory, f'{name}_prepared.sdf')
            if os.path.exists(out_file):
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(os.path.join(directory, f'{name}_prepared.sdf'),
                                                            sanitize=False, removeHs=False)
                for mol in supp:
                    if mol:
                        variant = mol.GetPropsAsDict()['s_lp_Variant']
                        if ':' in variant:
                            variant = variant.split(':')[1]
                        variant = variant.split('-')[1]
                        variants[name].append(variant)
                        out_split_file = os.path.join(directory, f'{name}-{variant}_prepared.sdf')
                        variant_files.append(out_split_file)
                        w = Chem.rdmolfiles.SDWriter(out_split_file)
                        w.write(mol)
                        w.flush()
                        w.close()
                        if self.logger: self.logger.debug(f'Split {name} -> {name}-{variant}')
                    else:
                        continue
        return variants, variant_files


class Epik(LigandPreparation):
    def __init__(self, dask_client=None, timeout: float = 120.0, logger=None, RDKit_stereoisomers: bool = True, max_stereo: int = 16):
        """
        Initialize Epik for ligand preparation, by default only enumerates the most likely protomer/tautomer and then enumerates stereoisomers.
        :param dask_client: Scheduler address for dask parallelization
        :param timeout: Timeout for subprocess before killing
        :param logger: Currently used logger if present
        :param enumerate_stereoisomers: Use RDKit to enumerate stereoisomers
        :param pH: pH at which to protonate molecules at
        :param pHt: pH Tolerance
        :param bff: Which forcefield to use, 14=OPLS3_2005 or 16=OPLS3e [14, 16]
        :param s: Maximum number of stereoisomers to generate, default is 32 but this is quite inefficient
        """
        super().__init__(dask_client=dask_client, timeout=timeout, logger=logger)
        self.RDKit_stereoisomers = RDKit_stereoisomers
        self.max_stereo = max_stereo
        # Setup env
        try:
            self.env = os.path.join(os.environ['SCHRODINGER'], 'epik')
            self.convert_env = os.path.join(os.environ['SCHRODINGER'], 'utilities', 'structconvert')
        except KeyError:
            raise KeyError("Please ensure you have a Schrodinger installation and license")           

    @staticmethod
    def enumerate_stereoisomers(smi: str, name: str, directory: os.PathLike, epik_env: str, convert_env: str):
        """
        For a given smiles string, and file name, enumerate stereoisomers and write out sdf and provide Epik command.
        :param smi: A single SMILES string
        :param name: File name e.g., <batch>_<batch_idx>
        :param directory: Working directory
        :return: (Name, Variants, Preparation Command)
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Add Hs
            mol = Chem.AddHs(mol)
            # Enumerate stereoisomers
            possible_stereoisomers = GetStereoisomerCount(mol)
            #logger.debug(f'{name}: {GetStereoisomerCount(mol)} possible stereoisomers')
            # Also embeds molecules
            opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, maxIsomers=16)
            stereoisomers = list(EnumerateStereoisomers(mol, options=opts))
            #logger.debug(f'{name}: {len(stereoisomers)} enumerated unique stereoisomers')
            variants = []
            out_paths = []
            commands = []
            for variant, iso in enumerate(stereoisomers):
                try:
                    Chem.EmbedMolecule(iso)
                except:
                    continue
                variants.append(variant)
                # Write to sdf
                sdf_in = os.path.join(directory, f'{name}-{variant}_isomers.sdf')
                mae_in = os.path.join(directory, f'{name}-{variant}_isomers.mae')
                sdf_out = os.path.join(directory, f'{name}-{variant}_prepared.sdf')
                mae_out = os.path.join(directory, f'{name}-{variant}_prepared.mae')
                out_paths.append(sdf_out)
                w = Chem.rdmolfiles.SDWriter(sdf_in)
                w.write(iso)
                w.flush()
                w.close()
                cmd = " ".join([convert_env, sdf_in, mae_in, ";",
                                    epik_env, f"-imae {mae_in}", f"-omae {mae_out}", "-ph 7.4", "-ms 1",
                                    "-WAIT", "-NOJOBID", ";",
                                    convert_env, mae_out, sdf_out])
                commands.append(cmd)
            assert isinstance(commands, list)
            return name, variants, commands, out_paths
        else:
            return name, [], [], []

    def prepare(self, smiles: list, directory: os.PathLike, file_names: list):
        """
        Use RDKit and Epik to prepare molecules.
        :param smiles: List of SMILES strings
        :param directory: Output directory
        :param file_names: File names for SMILES
        :return: (dictionary of variants, file_paths)
        """
        # Initialize subprocess
        if self.logger: self.logger.debug('Epik called')
        p = timedSubprocess(timeout=self.timeout, shell=True).run  # Use shell because ';'

        # Run commands either using Dask or sequentially
        if self.cluster:
            futures = [self.cluster.submit(self.enumerate_stereoisomers, smi, name, directory, epik_env=self.env, convert_env=self.convert_env)
                        for smi, name in zip(smiles, file_names)]
            results = self.cluster.gather(futures)
            variants = {}
            variant_files = []
            epik_commands = []
            for n, v, c, op in results:
                assert isinstance(c, list)
                variants[n] = v
                variant_files += op
                epik_commands += c
                #if c is not None:
                #    epik_commands.append(c)
            #variants = {n: v for n, v, c, op in results}
            #variant_files = [op for n, v, c, op in results]
            #epik_commands = [c for n, v, c, op in results if c is not None]
            futures = self.cluster.map(p, epik_commands)
            _ = self.cluster.gather(futures)

        else:
            results = [self.enumerate_stereoisomers(smi, name, directory, epik_env=self.env, convert_env=self.convert_env)
                        for smi, name in zip(smiles, file_names)]
            variants = {}
            variant_files = []
            epik_commands = []
            for n, v, c, op in results:
                variants[n] = v
                variant_files += op
                epik_commands += c
                #if c is not None:
                #    epik_commands.append(c)
            #variants = {n: v for n, v, c, op in results}
            #epik_commands = [c for n, v, c, op in results if c is not None]
            _ = [p(command) for command in epik_commands]
        
        if self.logger: self.logger.debug('Epik finished')
        return variants, variant_files


class Moka(LigandPreparation):
    def __init__(self, dask_client=None, timeout: float = 120.0, logger=None):
        """
        Initialize Moka/Corina ligand preparation
        :param dask_client: Scheduler address for dask parallelization
        :param timeout: Timeout for subprocess before killing
        :param logger: Currently used logger if present
        """
        super().__init__(dask_client=dask_client, timeout=timeout, logger=logger)
        # Setup env
        self.moka_env = subprocess.run(args=['which', 'blabber_sd'],
                                        stdout=subprocess.PIPE).stdout.decode().strip('\n')
        assert self.moka_env != '', "Please ensure you have a Moka installation and license"
        self.corina_env = subprocess.run(args=['which', 'corina'],
                                            stdout=subprocess.PIPE).stdout.decode().strip('\n')
        assert self.corina_env != '', "Please ensure you have a Corina installation and license"

    def prepare(self, smiles, directory, file_names):
        """
        Call Moka and Corina to prepare molecules.
        :param smiles: List of SMILES strings
        :param directory: Output directory
        :param file_names: File names for SMILES
        :return: (dictionary of variants, file_paths)
        """
        # Write out SMILES and Generate CLI commands
        moka_commands = []
        corina_commands = []
        for smi, name in zip(smiles, file_names):
            sdf_in = os.path.join(directory, f'{name}.sdf')
            sdf_moka = os.path.join(directory, f'{name}_moka.sdf')
            sdf_corina = os.path.join(directory, f'{name}_corina.sdf')
            mol = get_mol(smi)
            if mol is not None:
                # Have to write as sdf as rdkit has no mol2 functionality
                with Chem.rdmolfiles.SDWriter(sdf_in) as w:
                    w.write(mol)
                # blabber_sd -t 0.2 -p=7.4 --explicit-h=none -o <output_file.sdf> <input_file.mol2>
                moka_commands.append(f'{self.moka_env} -t 20 -p 7.4 --explicit-h=none -o {sdf_moka} {sdf_in}')
                # corina -d stergen,msi=16,wh,preserve <input_file> <output_file>
                corina_commands.append(f'{self.corina_env} -d stergen,msi=16,wh,preserve {sdf_moka} {sdf_corina}')

        # Initialize subprocesses
        if self.logger: self.logger.debug('Moka called')
        p = timedSubprocess(timeout=self.timeout, shell=False).run
        # Run commands either using Dask or sequentially
        if self.cluster:
            futures = self.cluster.map(p, moka_commands)
            _ = self.cluster.gather(futures)
            if self.logger: self.logger.debug('Moka finished')
            futures = self.cluster.map(p, corina_commands)
            _ = self.cluster.gather(futures)
            if self.logger: self.logger.debug('Corina finished')

        else:
            _ = [p(command) for command in moka_commands]
            if self.logger: self.logger.debug('Moka finished')
            _ = [p(command) for command in corina_commands]
            if self.logger: self.logger.debug('Corina finished')

        # Read in corina output files and split to individual variants
        variants = {name: [] for name in file_names}
        variant_files = []
        for name in file_names:
            out_file = os.path.join(directory, f'{name}_corina.sdf')
            if os.path.exists(out_file):
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(os.path.join(directory, f'{name}_corina.sdf'),
                                                            sanitize=False, removeHs=False)
                for variant, mol in enumerate(supp):
                    if mol:
                        variants[name].append(variant)
                        out_path = os.path.join(directory, f'{name}-{variant}_prepared.sdf')
                        variant_files.append(out_path)
                        with Chem.SDWriter(out_path) as w:
                            w.write(mol)
                            if self.logger: self.logger.debug(f'Split {name} -> {name}-{variant}')
                    else:
                        continue
        return variants, variant_files


class GypsumDL(LigandPreparation):
    """
    Makes use of and adapted from Gypsum-DL under Apache 2.0 license
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0358-3
        https://durrantlab.pitt.edu/gypsum-dl/
    """
    def __init__(self, logger=None, n_jobs: int = -1, pH: float = 7.0, pHt: float = 1.0):
        """
        Initialize LigPrep ligand preparation
        :param dask_client: Scheduler address for dask parallelization
        :param timeout: Timeout for subprocess before killing
        :param logger: Currently used logger if present
        :param n_jobs: Number of multiprocessing cores to use
        :param pH: pH at which to protonate molecules at
        :param pHt: pH Tolerance
        """
        # Note here we use GypsumDL' in-built parallelizer
        super().__init__(logger=logger)
        self.gypsum_params = {
            "source": "",
            "output_folder": "./",
            "separate_output_files": True,
            "add_pdb_output": False,
            "add_html_output": False,
            "num_processors": n_jobs,
            "start_time": 0,
            "end_time": 0,
            "run_time": 0,
            "min_ph": pH,
            "max_ph": pH,
            "pka_precision": pHt,
            "thoroughness": 1,
            "max_variants_per_compound": 8,
            "second_embed": False,
            "2d_output_only": False,
            "skip_optimize_geometry": True,
            "skip_alternate_ring_conformations": True,
            "skip_adding_hydrogen": False,
            "skip_making_tautomers": False,
            "skip_enumerate_chiral_mol": False,
            "skip_enumerate_double_bonds": False,
            "let_tautomers_change_chirality": False,
            "use_durrant_lab_filters": True,
            "job_manager": "multiprocessing",
            "cache_prerun": False,
            "test": False,
        }
        self.gypsum_params["Parallelizer"] = Parallelizer(
            self.gypsum_params["job_manager"], self.gypsum_params["num_processors"], True
        )

    
    def prepare(self, smiles: list, directory: os.PathLike, file_names: list):
        """
        Prepare smiles using GypsumDL's open-source ligand preparation protocol.
        :param smiles: List of SMILES strings
        :param directory: Output directory
        :param file_names: File names for SMILES
        :return: (dictionary of variants, file_paths)
        """
        # Load SMILES data
        smiles_data = [(smi, name, {}) for smi, name in zip(smiles, file_names)]

        # Make the molecule containers.
        contnrs = []
        for i in range(0, len(smiles_data)):
            smiles, name, props = smiles_data[i]
            new_contnr = MolContainer(smiles, name, i, props)
            contnrs.append(new_contnr)

        # Remove None types from failed conversion
        contnrs = [x for x in contnrs if x.orig_smi_canonical != None]

        # Prepare including protonation, tautomers and stereoenumeration
        prepare_smiles(contnrs, self.gypsum_params)

        # Convert the processed SMILES strings to 3D.
        prepare_3d(contnrs, self.gypsum_params)

        # Add in name and unique id to each molecule.
        add_mol_id_props(contnrs)

        # Save the output.
        variants = {name: [] for name in file_names}
        variant_files = []
        for i, contnr in enumerate(contnrs):
            # First of all remove duplicates
            contnr.remove_identical_mols_from_contnr()
            for v, m in enumerate(contnr.mols):
                if m:
                    variants[contnr.name].append(v)
                    m.load_conformers_into_rdkit_mol()
                    out_path = os.path.join(directory, f'{contnr.name}-{v}_prepared.sdf')
                    variant_files.append(out_path)
                    w = Chem.SDWriter(out_path)
                    w.write(m.rdkit_mol)
                    w.flush()
                    w.close()
                    if self.logger: self.logger.debug(f'Written {contnr.name}-{v}')
        return variants, variant_files

ligand_preparation_protocols = [LigPrep, Epik, Moka, GypsumDL]