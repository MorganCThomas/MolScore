
import os
import glob
import gzip
import logging

from openeye import oechem
from dask.distributed import Client
from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions.rocs import ROCS
from molscore.scoring_functions.utils import timedSubprocess

logger = logging.getLogger('glide')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class GlideDock:
    """
    Score structures using Glide docking score, including ligand preparation with LigPrep
    """
    return_metrics = ['r_i_docking_score', 'r_i_glide_ligand_efficiency', 'r_i_glide_ligand_efficiency_sa',
                      'r_i_glide_ligand_efficiency_ln', 'r_i_glide_gscore', 'r_i_glide_lipo',
                      'r_i_glide_hbond', 'r_i_glide_metal', 'r_i_glide_rewards', 'r_i_glide_evdw',
                      'r_i_glide_ecoul', 'r_i_glide_erotb', 'r_i_glide_esite', 'r_i_glide_emodel',
                      'r_i_glide_energy', 'r_i_glide_rmsd_to_input']

    def __init__(self, prefix: str, glide_template: os.PathLike, cluster: str = None,
                 timeout: float = 120.0, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param glide_template: Path to a template docking file (.in)
        :param cluster: Address to Dask scheduler for parallel processing via dask
        :param timeout: Timeout (seconds) before killing an individual docking simulation
        :param kwargs:
        """
        # Read in glide template (.in)
        with open(glide_template, 'r') as gfile:
            self.glide_options = gfile.readlines()
        # Make sure output file type is sdf
        self.glide_options = self.modify_glide_in(self.glide_options, 'POSE_OUTTYPE', 'ligandlib_sd')

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.glide_metrics = GlideDock.return_metrics
        self.glide_env = os.path.join(os.environ['SCHRODINGER'], 'glide')
        self.ligprep_env = os.path.join(os.environ['SCHRODINGER'], 'ligprep')
        self.timeout = float(timeout)
        self.cluster = cluster
        if self.cluster is not None:
            self.client = Client(self.cluster)
        self.variants = None
        self.docking_results = None

    @staticmethod
    def modify_glide_in(glide_in: str, glide_property: str, glide_value: str):
        """
        Convenience function to insert / overwrite certain .in file properties and values
        :param glide_in: A string of the .in file
        :param property: Property to be changed (e.g. POSE_OUTTYPE)
        :param value: Value to change to (e.g. ligandlib_sd)
        :return: Modified glide_in
        """
        # If property is already present, replace value
        if any([True if glide_property in line else False for line in glide_in]):
            for i in range(len(glide_in)):
                if glide_property in glide_in[i]:
                    glide_in[i] = f'{glide_property}   {glide_value}\n'
                    break
        # Otherwise insert it before newline (separates properties from features)
        elif any([True if line == '\n' else False for line in glide_in]):
            for i in range(len(glide_in)):
                if glide_in[i] == '\n':
                    glide_in.insert(i, f'{glide_property}   {glide_value}\n')
                    break
        # Otherwise just stick it on the end of the file
        else:
            glide_in.append(f'{glide_property}   {glide_value}\n')

        return glide_in

    def run_ligprep(self, smiles: list):
        """
        Call ligprep to prepare molecules.
        :param smiles: List of SMILES strings
        """
        ligprep_commands = []
        # Write out smiles to sdf files and prepare ligprep commands
        for smi, name in zip(smiles, self.file_names):
            smi_in = os.path.join(self.directory, f'{name}.smi')
            sdf_out = os.path.join(self.directory, f'{name}_ligprep.sdf')

            with open(smi_in, 'w') as f:
                f.write(smi)

            command = " ".join((self.ligprep_env,
                                f"-ismi {smi_in}",
                                f"-osd {sdf_out}",
                                "-ph 7.0",
                                "-pht 1.0",
                                "-bff 16",
                                "-s 8",
                                "-epik",
                                "-WAIT",
                                "-NOJOBID"))
            ligprep_commands.append(command)

        # Initialize subprocess
        logger.debug('LigPrep called')
        p = timedSubprocess(timeout=self.timeout).run

        # Run commands either using Dask or sequentially
        if self.cluster is not None:
            futures = self.client.map(p, ligprep_commands)
            _ = self.client.gather(futures)
        else:
            _ = [p(command) for command in ligprep_commands]
        logger.debug('LigPrep finished')
        return self

    @property
    def split_sdf(self):
        """
        Split ligprep output sdf so that each variant can be independently run.
        """
        # Read in ligprep output files and split to individual variants
        self.variants = {name: [] for name in self.file_names}
        for name in self.file_names:
            out_file = os.path.join(self.directory, f'{name}_ligprep.sdf')

            if os.path.exists(out_file):
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(os.path.join(self.directory, f'{name}_ligprep.sdf'),
                                                            sanitize=False, removeHs=False)

                for mol in supp:
                    if mol:
                        variant = mol.GetPropsAsDict()['s_lp_Variant']
                        if ':' in variant:
                            variant = variant.split(':')[1]
                        variant = variant.split('-')[1]
                        self.variants[name].append(variant)
                        w = Chem.rdmolfiles.SDWriter(os.path.join(self.directory, f'{name}-{variant}_ligprep.sdf'))
                        w.write(mol)
                        w.flush()
                        w.close()
                        logger.debug(f'Split {name} -> {name}-{variant}')
                    else:
                        continue

        return self

    def run_glide(self):
        """
        Write GLIDE new input files and submit each to Glide
        """
        glide_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                # Set some file paths
                glide_in = self.glide_options.copy()
                # Change glide_in file
                glide_in = self.modify_glide_in(glide_in,
                                                'LIGANDFILE',
                                                os.path.join(self.directory, f'{name}-{variant}_ligprep.sdf'))
                glide_in = self.modify_glide_in(glide_in,
                                                'OUTPUTDIR',
                                                os.path.join(self.directory))

                # Write new input file (.in)
                with open(os.path.join(self.directory, f'{name}-{variant}.in'), 'wt') as f:
                    [f.write(line) for line in glide_in]

                # Prepare command line command
                command = self.glide_env + ' -WAIT -NOJOBID -NOLOCAL ' + \
                          os.path.join(self.directory, f'{name}-{variant}.in')
                glide_commands.append(command)

        # Initialize subprocess
        logger.debug('Glide called')
        p = timedSubprocess(timeout=self.timeout).run

        if self.cluster is not None:
            futures = self.client.map(p, glide_commands)
            _ = self.client.gather(futures)
        else:
            _ = [p(command) for command in glide_commands]
        logger.debug('Glide finished')
        return self

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
            docking_result = {'smiles': smi}

            # For each variant
            for variant in self.variants[name]:
                out_file = os.path.join(self.directory, f'{name}-{variant}_lib.sdfgz')

                if os.path.exists(out_file):

                    # Try to load it in, and grab the score
                    try:
                        with gzip.open(out_file) as f:
                            glide_out = Chem.ForwardSDMolSupplier(f)

                            for mol in glide_out:  # should just be one
                                dscore = mol.GetPropsAsDict()['r_i_docking_score']

                                # If molecule doesn't have a score yet append it and the variant
                                if best_score[name] is None:
                                    best_score[name] = dscore
                                    best_variants[i] = f'{name}-{variant}'
                                    docking_result.update({f'{self.prefix}_' + k: v
                                                           for k, v in mol.GetPropsAsDict().items()
                                                           if k in self.glide_metrics})
                                    logger.debug(f'Docking score for {name}-{variant}: {dscore}')

                                # If docking score is better change it...
                                elif dscore < best_score[name]:
                                    best_score[name] = dscore
                                    best_variants[i] = f'{name}-{variant}'
                                    docking_result.update({f'{self.prefix}_' + k: v
                                                           for k, v in mol.GetPropsAsDict().items()
                                                           if k in self.glide_metrics})
                                    logger.debug(f'Found better {name}-{variant}: {dscore}')

                                # Otherwise ignore
                                else:
                                    pass

                    # If parsing the molecule threw an error and nothing stored, append 0
                    except:
                        logger.debug(f'Error processing {name}-{variant}_lib.sdfgz file')
                        if best_score[name] is None:  # Only if no other score for prefix
                            best_variants[i] = f'{name}-{variant}'
                            docking_result.update({f'{self.prefix}_' + k: 0.0 for k in self.glide_metrics})
                            logger.debug(f'Returning 0.0 unless a successful variant is found')

                # If path doesn't exist and nothing store, append 0
                else:
                    logger.debug(f'{name}-{variant}_lib.sdfgz does not exist')
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f'{name}-{variant}'
                        docking_result.update({f'{self.prefix}_' + k: 0.0 for k in self.glide_metrics})
                        logger.debug(f'Returning 0.0 unless a successful variant is found')

            # Add best variant information to docking result
            docking_result.update({f'{self.prefix}_best_variant': best_variants[i]})
            self.docking_results.append(docking_result)

        logger.debug(f'Best scores: {best_score}')
        if return_best_variant:
            logger.debug(f'Returning best variants: {best_variants}')
            return best_variants

        return self

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.cluster is None):
            parallel = False

        keep_poses = [f'{k}_lib.sdfgz' for k in keep]
        logger.debug(f'Keeping pose files: {keep_poses}')
        del_files = []
        for name in self.file_names:
            # Grab files
            files = glob.glob(os.path.join(self.directory, f'{name}*'))
            logger.debug(f'Glob found {len(files)} files')

            if len(files) > 0:
                try:
                    files = [file for file in files
                             if not ("log.txt" in file) and not any([p in file for p in keep_poses])]

                    if parallel:
                        [del_files.append(file) for file in files]
                    else:
                        [os.remove(file) for file in files]
                # No need to stop if files can't be found and deleted
                except FileNotFoundError:
                    logger.debug('File not found.')
                    pass
                
        if parallel:
            futures = self.client.map(os.remove, del_files)
            _ = self.client.gather(futures)
        return self

    def __call__(self, smiles: list, directory: str, file_names: list, **kwargs):
        """
        Calculate scores for GlideDock
        :param smiles: List of SMILES strings
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step

        # Create log directory
        self.directory = os.path.join(os.path.abspath(directory), 'GlideDock', step)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_names = file_names
        self.docking_results = []  # make sure no carry over

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f'{step}_log.txt'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Refresh Dask every few hundred iterations
        if self.cluster is not None:
            if int(step) % 250 == 0:
                self.client.restart()

        # Run protocol
        self.run_ligprep(smiles=smiles)
        self.split_sdf  # Catch any erroneous smiles with no output ligprep file
        self.run_glide()
        best_variants = self.get_docking_scores(smiles=smiles, return_best_variant=True)

        # Cleanup
        self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None
        
        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results


class GlideDockFromROCS(GlideDock, ROCS):
    """
    Score structures based on Glide docking score with LigPrep ligand preparation,
     but using ROCS to align to a reference molecule and score in place
    """
    return_metrics = GlideDock.return_metrics + ROCS.return_metrics

    def __init__(self, prefix: str, glide_template: os.PathLike, ref_file: os.PathLike, cluster: str = None,
                 timeout: float = 120.0, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param glide_template: Path to a template docking file (.in)
        :param ref_file: Path to reference file to overlay query to (.pdb)
        :param cluster: Address to Dask scheduler for parallel processing via dask
        :param timeout: Timeout (seconds) before killing an individual docking simulation
        :param kwargs:
        """
        GlideDock.__init__(self, prefix=prefix, glide_template=glide_template, cluster=cluster, timeout=timeout)
        ROCS.__init__(self, prefix=prefix, ref_file=ref_file)

        self.prefix = prefix.replace(" ", "")
        # Make sure glide template contains 'mininplace' method
        self.glide_options = self.modify_glide_in(self.glide_options, 'DOCKING_METHOD', 'mininplace')

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
        self.directory = os.path.join(os.path.abspath(directory), 'GlideDock', step)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_names = file_names
        self.docking_results = []

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f'{step}_log.txt'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Refresh Dask every few hundred iterations
        if self.cluster is not None:
            if int(step) % 250 == 0:
                self.client.restart()

        # Prepare ligands
        self.run_ligprep(smiles)

        # Read sdf files and run ROCS and write for docking
        results = []
        rocs_results = {}
        self.variants = {name: [] for name in self.file_names}
        for name in self.file_names:
            out_file = os.path.join(self.directory, f'{name}_ligprep.sdf')
            if os.path.exists(out_file):
                supp = Chem.rdmolfiles.ForwardSDMolSupplier(os.path.join(self.directory, f'{name}_ligprep.sdf'))
                for mol in supp:
                    if mol:
                        variant = mol.GetPropsAsDict()['s_lp_Variant'].split("-")[1]
                        self.variants[name].append(variant)
                        self.setup_smi(Chem.MolToSmiles(mol, isomericSmiles=True))
                        self.run_omega()

                        # Hack to catch 'valid molecules' that have no coordinates after omega init
                        if len(self.fitmol.GetCoords()) == 0:
                            rocs_results[f'{name}_{variant}'] = {f'{self.prefix}_{m}': 0.0 for m in self.rocs_metrics}
                            continue

                        # Run ROCS and write each variants best overlay
                        self.run_ROCS()
                        rocs_results[f'{name}_{variant}'] = {m: getattr(self.rocs_results, m)()
                                                             for m in self.rocs_metrics}
                        self.get_best_overlay()
                        ofs = oechem.oemolostream(os.path.join(self.directory, f'{name}-{variant}_ligprep.sdf'))
                        if oechem.OEAddExplicitHydrogens(self.best_overlay):
                            oechem.OEWriteMolecule(ofs, self.best_overlay)
                        ofs.close()
                        logger.debug(f'Split and aligned {name} -> {name}-{variant}')

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


#class GlideDockFromGlide(GlideDock):
    # TODO implement inplace glide docking from previous dock
#    raise NotImplementedError



