import logging
import os
import time
from typing import Union
from openbabel import pybel as pb
import numpy as np
import signal
from molscore.scoring_functions.utils import Pool
from molscore.scoring_functions.utils import timedFunc2
from rdkit import Chem

from ccdc import io
from ccdc import conformer
from ccdc.molecule import Molecule as ccdcMolecule

from hsr import pre_processing as pp
from hsr import fingerprint as fp
from hsr import similarity as sim
from hsr.utils import PROTON_FEATURES

logger = logging.getLogger("HSR")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class TimeoutException(Exception):
    pass

# Set up a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the signal handler for alarm
signal.signal(signal.SIGALRM, timeout_handler)


def get_array_from_pybelmol(pybelmol):
    # Iterate over the atoms in the molecule
    atom_array = []
    for atom in pybelmol.atoms:
        # Append the coordinates and the atomic number of the atom (on each row)
        atom_array.append([atom.coords[0], atom.coords[1], atom.coords[2], atom.atomicnum])    
    # Centering the data
    atom_array -= np.mean(atom_array, axis=0)
    return atom_array

def get_array_from_ccdcmol(ccdcmol):
    # Iterate over the atoms in the molecule
    atom_array = []
    for atom in ccdcmol.atoms:
        # Append the coordinates and the atomic number of the atom (on each row)
        atom_array.append([atom.coordinates[0], atom.coordinates[1], atom.coordinates[2], atom.atomic_number])    
    # Centering the data
    atom_array -= np.mean(atom_array, axis=0)
    return atom_array

class HSR:
    """
    HSR (hyper-shape recognition) similarity measure
    """
    
    return_metrics = ["HSR_score"]
    
    def __init__(
        self,
        prefix: str,
        ref_molecule: os.PathLike,
        generator: str = 'ccdc',
        n_jobs: int = 1,    
        timeout: int = 10,
        save_files: bool = False,
        
    ):
        """
        Initialize HSR similarity measure
        
        :param prefix: Prefix to identify scoring function instance
        :param ref_molecule: Reference molecule file path
        :param generator: Generator of 3D coordinates, also package used to open the reference molecule file.
        options: 'ccdc', 'obabel', 'rdkit' 
        :param n_jobs: Number of parallel jobs (number of molecules to score in parallel)
        :param timeout: Timeout value for the 3D generation process (in seconds)
        """
        
        self.prefix = prefix.strip().replace(" ", "_")
        self.ref_molecule = pp.read_mol_from_file(ref_molecule)
        if self.ref_molecule is None:
            raise ValueError("Reference molecule is None. Check if the extension of the file is managed by HSR")
        self.generator = generator
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.save_files = save_files
        if self.generator == 'ccdc':
            #Read molecule from file 
            ref_mol = io.MoleculeReader(ref_molecule)[0]
            ref_mol_array = get_array_from_ccdcmol(ref_mol)
            self.ref_mol_fp = fp.generate_fingerprint_from_data(ref_mol_array)
        elif self.generator == 'obabel':
            #Read molecule from file (sdf)
            ref_mol = next(pb.readfile("sdf", ref_molecule))
            ref_mol_array = get_array_from_pybelmol(ref_mol)
            self.ref_mol_fp = fp.generate_fingerprint_from_data(ref_mol_array)
        elif self.generator == 'rdkit':
            # Generate fingerprint directly from the molecule
            self.ref_mol_fp = fp.generate_fingerprint_from_molecule(self.ref_molecule, PROTON_FEATURES)
    
    def save_mols_to_file(self, results: dict, directory: str, file_names: list):
        """
        Save molecules to SDF files
        """
        for r, name in zip(results, file_names):
            file_path = os.path.join(directory, name + ".sdf")
            try:
                mol_sdf = r[f"3d_mol"]
                score = r[f"{self.prefix}_HSR_score"]
                if score > 0.0:
                    successes += 1
                if mol_sdf:
                    with open(file_path, "w") as f:
                        f.write(mol_sdf)
                        f.write(f"\n> <SMILES>\n{r['smiles']}\n\n$$$$\n")
                        f.write(f"\n> <HSR_score>\n{score}\n\n$$$$\n")
                        
            except KeyError:
                continue
    
    def get_mols_3D(self, smiles: str):
        """
        Generate 3D coordinates for a SMILES
        
        :param smiles: SMILES string
        :return: 
        mol_sdf: 3D coordinates in SDF format
        mol_3d: 3D molecule object
        """
        try:
            # Set the alarm for the timeout
            signal.alarm(self.timeout) 
            if self.generator == "obabel": 
                mol_3d = pb.readstring("smi", smiles)
                mol_3d.addh()
                mol_3d.make3D()
                mol_3d.localopt()
                mol_sdf = mol_3d.write("sdf")
            elif self.generator == "ccdc":
                conformer_generator = conformer.ConformerGenerator(nthreads = self.n_jobs)
                mol = ccdcMolecule.from_string(smiles)
                conformer_generator.settings.max_conformers = 1
                conf = conformer_generator.generate(mol)
                mol_3d = conf.hits[0].molecule
                mol_sdf = mol_3d.to_string("sdf")
            elif self.generator == "rdkit":
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                Chem.AllChem.EmbedMolecule(mol)
                Chem.AllChem.MMFFOptimizeMolecule(mol)
                mol_sdf = Chem.MolToMolBlock(mol)
                mol_3d = mol
            signal.alarm(0)
        except TimeoutException:
            logger.warning(f"Timeout occurred during 3D generation for {smiles}")
            mol_3d, mol_sdf = None, None
        except Exception as e:
            logger.error(f"Error generating 3D coordinates for {smiles}: {e}")
            mol_3d, mol_sdf = None, None
            
        return mol_3d, mol_sdf

    def score_mol(self, mol: Union[str, pb.Molecule, Chem.Mol, np.ndarray]):
        """
        Calculate HSR similarity between a molecule and the reference molecule
        :param mol: SMILES string, molecule object (rdkit, pybel, ccdc) or numpy array
        :return: dict with the HSR similarity score
        """
        result = {}
        try:
            #Check what object the molecule is:
            
            #The molecule is a SMILES string
            if isinstance(mol, str):
                result = {"molecule": mol} 
                molecule, mol_sdf = self.get_mols_3D(mol)
                # Handle exceptions in 3D generation gracefully
                if molecule is None and mol_sdf is None:
                    raise ValueError(f"Error generating 3D coordinates for {mol}")
                result.update({f'3d_mol': mol_sdf})
                if isinstance(molecule, pb.Molecule):
                    mol_array = get_array_from_pybelmol(molecule)
                    mol_fp = fp.generate_fingerprint_from_data(mol_array)
                elif isinstance(molecule, ccdcMolecule):
                    mol_array = get_array_from_ccdcmol(molecule)
                    mol_fp = fp.generate_fingerprint_from_data(mol_array)
                    
            #The molecule is a rdkit molecule
            elif isinstance(mol, Chem.Mol):
                result = {"molecule": mol}
                # Check if molecule is 3d, otherwise generate 3D coordinates
                if not mol.GetNumConformers():
                    raise Exception("Molecule is not 3D")
                mol_sdf = Chem.MolToMolBlock(mol)
                result.update({f'3d_mol': mol_sdf})
                mol_fp = fp.generate_fingerprint_from_molecule(mol, PROTON_FEATURES)
                
            #The molecule is a pybel molecule (obabel)
            elif isinstance(mol, pb.Molecule):
                result = {"molecule": mol}
                result.update({f'3d_mol': mol.write("sdf")})
                mol_array = get_array_from_pybelmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array)
               
            #The molecule is a ccdc molecule 
            elif isinstance(mol, ccdcMolecule):
                result = {"molecule": mol}
                mol_sdf = mol.to_string("sdf")
                mol_array = get_array_from_ccdcmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array)
                
            #The molecule is an np.array
            elif isinstance(mol, np.ndarray):
                result = {"molecule": mol}
                mol_fp = fp.generate_fingerprint_from_data(molecule, scaling='matrix')
                
        except Exception as e:
            result.update({
                f'molecule': mol,
                f'3d_mol': 0.0,
                f'{self.prefix}_HSR_score': 0.0
            })
            return result
            
        # Calculate the similarity
        try:
            sim_score = sim.compute_similarity_score(self.ref_mol_fp, mol_fp)
        except Exception as e:
            result.update({f'{self.prefix}_HSR_score': 0.0})
            logger.error(f"Error calculating HSR similarity for {mol}: {e}")
        if np.isnan(sim_score):
            sim_score = 0.0 
        result.update({f'{self.prefix}_HSR_score': sim_score})
        
        return result
    
    def score(self, molecules: list, directory, file_names, **kwargs):
        """
        Calculate the HSR similarity scores of a list of molecules to a reference molecule.
        :param molecules: List of molecules to score: SMILES, molecule objects (rdkit, pybel, ccdc) or numpy arrays
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names to save the molecules
        :return: List of dicts i.e. [{'molcule': mol, 'HSR_score': 'value', ...}, ...]
        """
        # Create directory
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_HSR", step
        )
        os.makedirs(directory, exist_ok=True)
        pfunc = timedFunc2(self.score_mol, self.timeout)
        # Use parallelisation only with obabel and rdkit (and SMILES strings)
        results = []
        if isinstance(molecules[0], str) and self.generator in {'obabel', 'rdkit'}:
            n_processes = min(self.n_jobs, len(molecules), os.cpu_count())
            with Pool(n_processes) as pool:
                async_results = []
                for mol in molecules:
                    async_result = pool.apply_async(pfunc, args=(mol,))
                    async_results.append(async_result)

                for i, async_result in enumerate(async_results):
                    try:
                        result = async_result.get()
                        if result is None:
                            logger.warning(f"Timeout occurred during scoring for molecule {molecules[i]}")
                            result = {
                                "molecule": molecules[i],
                                '3d_mol': 0.0,
                                f'{self.prefix}_HSR_score': 0.0,
                            }
                    except Exception as e:
                        # Handle any other exception
                        logger.error(f"Error processing molecule {molecules[i]}: {e}")
                        result = {
                            "molecule": molecules[i],
                            f'3d_mol': 0.0,
                            f'{self.prefix}_HSR_score': 0.0,
                        }
                    results.append(result)             
        else:
            # In all other cases, process molecules sequentially
            for i, mol in enumerate(molecules):
                start = time.time()
                result = pfunc(mol)
                if result is None:
                    logger.warning(f"Timeout occurred during scoring for molecule {mol}")
                    logger.info(f"Scored molecule {i} in {time.time() - start:.2f} s")
                    result = {
                        "molecule": mol,
                        '3d_mol': 0.0,
                        f'{self.prefix}_HSR_score': 0.0,
                    }
                else:
                    logger.info(f"Scored molecule {i} in {time.time() - start:.2f} s")
                results.append(result)
                
        # Check if in results there is the key '3d_mol' (we are not using ndarray, 
        # from whihc we cannot retrieve the 3D coordinates) 
        if self.save_files:
            if '3d_mol' in results[0].keys():
                self.save_mols_to_file(results, directory, file_names)
            else:
                logger.warning("3D coordinates not available for saving to file")
        
        # Remove 3d_mol key from results
        for r in results:
            if '3d_mol' in r.keys():
                del r['3d_mol']
        
        return results

            
    def __call__(self, directory, file_names, smiles=None, **kwargs):
        """
        Calculate the scores based on HSR similarity to reference molecules.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        if smiles is None:
            molecules =kwargs.get('molecular_inputs', [])
            if molecules is None:
                raise ValueError("Either 'smiles' or 'molecules' in kwargs must be provided.")
        else:
            molecules = smiles
            
        return self.score(molecules, directory=directory, file_names=file_names)