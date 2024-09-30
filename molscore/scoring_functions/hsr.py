import logging
import os
from typing import Union
from openbabel import openbabel as ob
from openbabel import pybel as pb
import numpy as np
from functools import partial
from molscore.scoring_functions.utils import Pool

from ccdc import conformer
from ccdc.molecule import Molecule

from hsr import pre_processing as pp
from hsr import  pca_transform as pca
from hsr import fingerprint as fp
from hsr import similarity as sim
from hsr.utils import PROTON_FEATURES

logger = logging.getLogger("HSR")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def get_array_from_obmol(obmol):
    # Iterate over the atoms in the molecule
    atom_array = []
    for atom in ob.OBMolAtomIter(obmol):
        # Append the coordinatess and the atomic number of the atom (on each row)
        atom_array.append([atom.GetX(), atom.GetY(), atom.GetZ(), atom.GetAtomicNum()])    
   
    # Centering the data
    atom_array = np.array(atom_array) - np.mean(atom_array, axis=0)
    return atom_array

def get_array_from_pybelmol(pybelmol):
    # Iterate over the atoms in the molecule
    atom_array = []
    for atom in pybelmol.atoms:
        # Append the coordinates and the atomic number of the atom (on each row)
        atom_array.append([atom.coords[0], atom.coords[1], atom.coords[2], atom.atomicnum])    
   
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
        generator: str,  
        n_jobs: int = 1,    
        #TODO: add parameteres to tweak the similairty measure
    ):
        """
        Initialize HSR similarity measure
        
        :param prefix: Prefix to identify scoring function instance
        :param ref_molecule: reference molecule file path
        :param generator: generator of 3D coordinates, ('obabel' is the only one supported for now, 'ccdc' will be added soon)
        :param n_jobs: number of parralel jobs (number of molecules to score in parallel)
        #TODO: add parameteres to tweak the similairty measure
        """
        
        self.prefix = prefix.strip().replace(" ", "_")
        self.ref_molecule = pp.read_mol_from_file(ref_molecule)
        if self.ref_molecule is None:
            raise ValueError("Reference molecule is None. Check the extesnsion of the file is managed by HSR")
        self.ref_mol_fp = fp.generate_fingerprint_from_molecule(self.ref_molecule, PROTON_FEATURES, scaling='matrix')
        self.generator = generator
        self.n_jobs = n_jobs
    
    def get_mols_3D(self, smiles: list):
        """
        Generate 3D coordinates for a list of SMILES
        :param smiles: list of SMILES
        :return: list of molecules with 3D coordinates
        """
        mols_3D = []
        if self.generator == "obabel": #TODO: add more generators
            for smile in smiles:
                mol = pb.readstring("smi", smile)
                mol.addh()
                mol.make3D()
                mol.localopt()
                mols_3D.append(mol)
        # elif self.generator == "ccdc":
        #     conformer_generator = conformer.ConformerGenerator()
        #     for smile in smiles:
        #         mol = Molecule.from_string(smile)
        #         conf = conformer_generator.generate(mol, hydrogens=False)
        #         mol_3d = conf.hits[0].molecule
        #         mols_3D.append(mol_3d)
            
        return mols_3D
    

    def score_smi(self, smi: str):
        
        result = {"smiles": smi}

        # Generate 3D coordinates
        try:
            
            if type(smi)==str: #The molecule is a SMILES string
                molecule = self.get_mols_3D([smi])[0]
                # Serialize molecule to SDF (to use in multiprocessing)
                # TODO: Thi step assume a obmol, it should be generalized to be agnostic to the molecule object
                mol_sdf = molecule.write("sdf")
                result.update({f'3d_mol': mol_sdf})
            elif type(smi).__name__=="Mol": #The molecule is a rdkit molecule
                mol_fp = fp.generate_fingerprint_from_molecule(molecule, PROTON_FEATURES, scaling='matrix')

        
            #Check what object the molecule is:
            # check if it is a obabel molecule
            if type(molecule).__name__ == "OBMol":
                mol_array = get_array_from_obmol(molecule)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
            # check if it is a pybel molecule
            elif type(molecule).__name__ == "Molecule":
                mol_array = get_array_from_pybelmol(molecule)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
            # check if it is an np.array
            elif type(molecule).__name__ == "ndarray":
                mol_fp = fp.generate_fingerprint_from_data(molecule, scaling='matrix')
                
        except Exception as e:
            logger.error(f"Error generating 3D coordinates for {smi}: {e}")
            result.update({
                f'3d_mol': 0.0,
                f'{self.prefix}_HSR_score': 0.0
            })
            return result
            

        # Calculate the similarity
        try:
            sim_score = sim.compute_similarity_score(self.ref_mol_fp, mol_fp)
        except Exception as e:
            result.update({f'{self.prefix}_HSR_score': 0.0})
            logger.error(f"Error calculating HSR similarity for {smi}: {e}")
        
        result.update({f'{self.prefix}_HSR_score': sim_score})
        return result
    
    
    def score(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on HSR similarity to reference molecules.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Create directory
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_HSR", step
        )
        os.makedirs(directory, exist_ok=True)
        # Prepare function for parallelization
        
        # pfunc = partial(
        #     self.score_smi,
        #     #TODO: add more parameters
        #     )
        
        # # Score individual smiles
        # with Pool(self.n_jobs) as pool:
        #     results = [r for r in pool.imap(pfunc, smiles)]
            
        # for loop (for debugging)
        results = []
        for smi in smiles:
            result = self.score_smi(smi)
            results.append(result)
            
        # Save mols
        for r, name in zip(results, file_names):
            file_path = os.path.join(directory, name + ".sdf")
            try:
                mol_sdf = r.pop(f"3d_mol")
                if mol_sdf:
                    with open(file_path, "w") as f:
                        f.write(mol_sdf)
                        # Save also the smiles under a field in the sdf file
                        # This is useful for further analysis
                        f.write(f"\n> <SMILES>\n{r['smiles']}\n\n$$$$\n")
                        
            except KeyError:
                continue
        return results
            
    def __call__(self, smiles: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on HSR similarity to reference molecules.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles, directory=directory, file_names=file_names)

        
        
   