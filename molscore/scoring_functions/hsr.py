import logging
import os
from typing import Union
from openbabel import openbabel as ob
from openbabel import pybel as pb
import numpy as np
from functools import partial
from molscore.scoring_functions.utils import Pool
from rdkit import Chem

from ccdc import io
from ccdc import conformer
from ccdc.molecule import Molecule as ccdcMolecule

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
        self.generator = generator
        self.n_jobs = n_jobs
        # TODO: Add the case when the there is no need for a generator and the molecule is already 3D
        # or the array is directly provided
        if self.generator == 'ccdc':
            #Read molecule from file 
            ref_mol = io.MoleculeReader(ref_molecule)[0]
            ref_mol_array = get_array_from_ccdcmol(ref_mol)
            self.ref_mol_fp = fp.generate_fingerprint_from_data(ref_mol_array, scaling='matrix')
        elif self.generator == 'obabel':
            #Read molecule from file (sdf)
            ref_mol = next(pb.readfile("sdf", ref_molecule))
            ref_mol_array = get_array_from_pybelmol(ref_mol)
            self.ref_mol_fp = fp.generate_fingerprint_from_data(ref_mol_array, scaling='matrix')
        else:
            #Default for now: rdkit
            #TODO: explicitly insert rdkit as a conformer generatore with the disclaimer that 
            # it cannot deal with organometallic molecules
            self.ref_mol_fp = fp.generate_fingerprint_from_molecule(self.ref_molecule, PROTON_FEATURES, scaling='matrix')
 
    def get_mols_3D(self, smile: str):
        """
        Generate 3D coordinates for a list of SMILES
        :param smile: SMILES string
        :return: 
        """
        if self.generator == "obabel": 
            mol_3d = pb.readstring("smi", smile)
            mol_3d.addh()
            mol_3d.make3D()
            mol_3d.localopt()
            mol_sdf = mol_3d.write("sdf")
        elif self.generator == "ccdc":
            conformer_generator = conformer.ConformerGenerator()
            mol = ccdcMolecule.from_string(smile)
            conf = conformer_generator.generate(mol)
            mol_3d = conf.hits[0].molecule
            mol_sdf = mol_3d.to_string("sdf")
        return mol_3d, mol_sdf

    def score_mol(self, mol: Union[str, pb.Molecule, Chem.Mol, np.ndarray]):
        """
        Calculate the HSR similarity score for a molecule
        :param mol: SMILES string, rdkit molecule or numpy array
        :return: dict with the HSR similarity score
        """

        # Generate 3D coordinates
        try:
            #Check what object the molecule is:
            
            #The molecule is a SMILES string
            if isinstance(mol, str):
                # TODO: This will have to become the molecule object or its identifier
                result = {"smiles": mol}
                # TODO: This instead has to become the smiles field of the molecule (if available)
                result.update({f"smiles_ph": mol})
                
                molecule, mol_sdf = self.get_mols_3D(mol)
                result.update({f'3d_mol': mol_sdf})
                if isinstance(molecule, pb.Molecule):
                    mol_array = get_array_from_pybelmol(molecule)
                    mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
                elif isinstance(molecule, ccdcMolecule):
                    mol_array = get_array_from_ccdcmol(molecule)
                    mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
            #The molecule is a rdkit molecule
            elif isinstance(mol, Chem.Mol):
                # If the conversion to smiles is successful save smiles in result
                #TODO: smae as above
                result = {"smiles": mol}
                try:
                    result.update({f"smiles_ph": Chem.MolToSmiles(mol)})
                except Exception as e:
                    logger.error(f"Error converting rdkit molecule to smiles: {e}")
                    result.update({f"smiles_ph": "N/A"})

                # Check if molecule is 3d, otherwise generate 3D coordinates
                if not mol.GetNumConformers():
                    mol = Chem.AddHs(mol)
                    Chem.AllChem.EmbedMolecule(mol)
                    Chem.AllChem.MMFFOptimizeMolecule(mol)
                
                mol_sdf = Chem.MolToMolBlock(mol)
                result.update({f'3d_mol': mol_sdf})
                mol_fp = fp.generate_fingerprint_from_molecule(mol, PROTON_FEATURES, scaling='matrix')
                
            #The molecule is a pybel molecule (obabel)
            elif isinstance(mol, pb.Molecule):
                # TODO: Same as above
                result = {"smiles": mol}
                try:
                    result.update({f"smiles_ph": mol.write("smi")})
                except Exception as e:
                    logger.error(f"Error converting pybel molecule to smiles: {e}")
                    result.update({f"smiles_ph": "N/A"})
                result.update({f'3d_mol': mol.write("sdf")})
                mol_array = get_array_from_pybelmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
               
            #The molecule is a ccdc molecule 
            elif isinstance(mol, ccdcMolecule):
                # TODO: Same as above
                result = {"smiles": mol}
                try: 
                    result.update({f"smiles_ph": mol.to_smiles()})
                except Exception as e:
                    logger.error(f"Error converting ccdc molecule to smiles: {e}")
                    result.update({f"smiles_ph": "N/A"})
                result.update({f'3d_mol': mol.to_string("sdf")})
                mol_array = get_array_from_ccdcmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
                
            #The molecule is an np.array
            elif isinstance(mol, np.ndarray):
                # TODO: Add the possibility to save the 'molecule' in a file
                # result = {"smiles": "N/A"}
                mol_fp = fp.generate_fingerprint_from_data(molecule, scaling='matrix')
                
        except Exception as e:
            logger.error(f"Error generating 3D coordinates for {mol}: {e}")
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
            logger.error(f"Error calculating HSR similarity for {mol}: {e}")
        # print(f"HSR score: {sim_score}")
        if np.isnan(sim_score):
            sim_score = 0.0 
        result.update({f'{self.prefix}_HSR_score': sim_score})
        return result
    
    
    def score(self, molecules: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on HSR similarity to reference molecules.
        :param molecules: List of molecules to score, can be SMILES strings, rdkit molecules or numpy arrays
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
        
        pfunc = partial(
            self.score_mol,
            #TODO: add more parameters
            )
        
        # Score individual smiles
        with Pool(self.n_jobs) as pool:
            results = [r for r in pool.imap(pfunc, molecules)]
            
        # # for loop (for easier debugging)
        # results = []
        # for mol in molecules:
        #     result = self.score_mol(mol)
        #     results.append(result)
            
        # Save mols
        for r, name in zip(results, file_names):
            file_path = os.path.join(directory, name + ".sdf")
            try:
                mol_sdf = r[f"3d_mol"]
                score = r[f"{self.prefix}_HSR_score"]
                if mol_sdf:
                    with open(file_path, "w") as f:
                        f.write(mol_sdf)
                        f.write(f"\n> <SMILES>\n{r['smiles']}\n\n$$$$\n")
                        f.write(f"\n> <HSR_score>\n{score}\n\n$$$$\n")
                        
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
        return self.score(molecules=smiles, directory=directory, file_names=file_names)

    