import logging
import os
import time
import signal
from typing import Union
from openbabel import pybel as pb
import numpy as np
from functools import partial
from molscore.scoring_functions.utils import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Empty
import multiprocessing
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
        generator: str,  #TODO: make generator optional
        n_jobs: int = 1,    
        timeout: int = 10 
        
    ):
        """
        Initialize HSR similarity measure
        
        :param prefix: Prefix to identify scoring function instance
        :param ref_molecule: reference molecule file path
        :param generator: generator of 3D coordinates, ('obabel' is the only one supported for now, 'ccdc' will be added soon)
        :param n_jobs: number of parallel jobs (number of molecules to score in parallel)
        :param timeout: Timeout value for the 3D generation process (in seconds)
        """
        
        self.prefix = prefix.strip().replace(" ", "_")
        self.ref_molecule = pp.read_mol_from_file(ref_molecule)
        if self.ref_molecule is None:
            raise ValueError("Reference molecule is None. Check the extesnsion of the file is managed by HSR")
        self.generator = generator
        self.n_jobs = n_jobs
        self.timeout = timeout
        # TODO: Add the case when there is no need for a generator and the molecule is already 3D
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
        elif self.generator == 'rdkit':
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
        try: 
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
        except Exception as e:
            logger.error(f"Error generating 3D coordinates for {smile}: {e}")
            mol_3d, mol_sdf = None, None
            
        return mol_3d, mol_sdf

    def score_mol(self, mol: Union[str, pb.Molecule, Chem.Mol, np.ndarray]):
        """
        Calculate the HSR similarity score for a molecule
        :param mol: SMILES string, rdkit molecule or numpy array
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
                    mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
                elif isinstance(molecule, ccdcMolecule):
                    mol_array = get_array_from_ccdcmol(molecule)
                    mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
                    
            #The molecule is a rdkit molecule
            elif isinstance(mol, Chem.Mol):
                result = {"molecule": mol}
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
                result = {"molecule": mol}
                result.update({f'3d_mol': mol.write("sdf")})
                mol_array = get_array_from_pybelmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
               
            #The molecule is a ccdc molecule 
            elif isinstance(mol, ccdcMolecule):
                result = {"molecule": mol}
                mol_array = get_array_from_ccdcmol(mol)
                mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
                
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
    
    def score_mol_worker(self, mol, result_queue):
        """
        Worker to invoke the scoring function and put the result in the 
        multiprocessing queue.
        """
        try:
            result = self.score_mol(mol)
            result_queue.put(result)  # Put the result into the queue
        except Exception as e:
            # In case of any error, put a default result in the queue
            result_queue.put({
                f"molecule": mol,
                f'3d_mol': 0.0,
                f'{self.prefix}_HSR_score': 0.0,
            })
    
    def score(self, molecules: list, directory, file_names, **kwargs):
        """
        Calculate the scores based on HSR similarity to reference molecules.
        """
        # Create directory
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_HSR", step
        )
        os.makedirs(directory, exist_ok=True)
        
        #TODO: Implement parallel processing able to:
        # 1. Handle timeouts
        # 2. Handle not pickable objects
        # 3. Handle zombie processes
        
        results = []
        for mol in molecules:
            result = self.score_mol(mol)
            results.append(result)
            
        # Save mols
        successes = 0
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



#  results = []
#         with ProcessPoolExecutor(max_workers=n_processes) as executor:
#             # Submit tasks and collect futures
#             futures = {executor.submit(pfunc, mol): mol for mol in molecules}

#             # Iterate over completed futures
#             for future in as_completed(futures):
#                 mol = futures[future]
#                 try:
#                     # The result will timeout after the given limit
#                     result = future.result(timeout=self.timeout)
#                 except TimeoutError:
#                     logger.error(f"Timeout occurred for molecule: {mol}")
#                     result = {
#                         "smiles": mol,
#                         f'3d_mol': 0.0,
#                         f'{self.prefix}_HSR_score': 0.0,
#                         'time_taken': self.timeout
#                     }
#                 except Exception as e:
#                     logger.error(f"Error processing molecule {mol}: {e}")
#                     result = {
#                         "smiles": mol,
#                         f'3d_mol': 0.0,
#                         f'{self.prefix}_HSR_score': 0.0,
#                         'time_taken': self.timeout
#                     }
#                 results.append(result)
            
            
        
        # results = []
        # with Pool(n_processes) as pool:
        
        #     # Submit tasks with apply_async and set a timeout for each scoring
        #     async_results = []
        #     for mol in molecules:
        #         async_result = pool.apply_async(pfunc, args=(mol,))
        #         async_results.append(async_result)

        #     # Collect results, applying timeout and handling it gracefully
        #     for i, async_result in enumerate(async_results):
        #         try:
        #             # Try to get the result with a timeout equal to the specified time limit
        #             result = async_result.get(timeout=self.timeout)
        #         except multiprocessing.TimeoutError:
        #             # Handle the timeout scenario by using default values
        #             # logger.error(f"Timeout occurred for molecule: {molecules[i]}")
        #             result = {
        #                 "smiles": molecules[i],
        #                 f'3d_mol': 0.0,
        #                 f'{self.prefix}_HSR_score': 0.0,
        #                 'time_taken': self.timeout
        #             }
        #         except Exception as e:
        #             # Handle any other exception
        #             logger.error(f"Error processing molecule {molecules[i]}: {e}")
        #             result = {
        #                 "smiles": molecules[i],
        #                 f'3d_mol': 0.0,
        #                 f'{self.prefix}_HSR_score': 0.0,
        #                 'time_taken': self.timeout
        #             }
        #         finally:
        #             results.append(result)
        
        
        # with Pool(n_processes) as pool:
        #     results = [r for r in pool.imap(pfunc, molecules)]
        
       
        
        
        # def score_mol(self, mol: Union[str, pb.Molecule, Chem.Mol, np.ndarray]):
        # """
        # Calculate the HSR similarity score for a molecule
        # :param mol: SMILES string, rdkit molecule, or numpy array
        # :return: dict with the HSR similarity score
        # """
        # start_time = time.time()
        # timeout = self.timeout  # The timeout value in seconds

        # result = {}
        # try:
        #     # Check if the timeout is exceeded
        #     def check_timeout():
        #         if time.time() - start_time > timeout:
        #             raise TimeoutError(f"Processing of molecule {mol} exceeded the allowed time of {timeout} seconds.")

        #     # ---- Start processing ----

        #     # The molecule is a SMILES string
        #     if isinstance(mol, str):
        #         check_timeout()  # Check timeout at key points
        #         result = {"smiles": mol}
        #         result.update({f"smiles_ph": mol})
                
        #         molecule, mol_sdf = self.get_mols_3D(mol)
        #         check_timeout()  # Check timeout again after a potentially time-consuming operation

        #         # Handle exceptions in 3D generation gracefully
        #         if molecule is None and mol_sdf is None:
        #             raise ValueError(f"Error generating 3D coordinates for {mol}")
        #         result.update({f'3d_mol': mol_sdf})

        #         if isinstance(molecule, pb.Molecule):
        #             mol_array = get_array_from_pybelmol(molecule)
        #             check_timeout()  # Check timeout after an array generation operation
        #             mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')
        #         elif isinstance(molecule, ccdcMolecule):
        #             mol_array = get_array_from_ccdcmol(molecule)
        #             check_timeout()  # Check timeout again
        #             mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')

        #     # The molecule is an RDKit molecule
        #     elif isinstance(mol, Chem.Mol):
        #         check_timeout()  # Check timeout at key points
        #         result = {"smiles": mol}
        #         try:
        #             result.update({f"smiles_ph": Chem.MolToSmiles(mol)})
        #         except Exception as e:
        #             logger.error(f"Error converting rdkit molecule to smiles: {e}")
        #             result.update({f"smiles_ph": "N/A"})

        #         # Check if molecule is 3D, otherwise generate 3D coordinates
        #         if not mol.GetNumConformers():
        #             mol = Chem.AddHs(mol)
        #             check_timeout()  # Check timeout before a potentially long operation
        #             Chem.AllChem.EmbedMolecule(mol)
        #             Chem.AllChem.MMFFOptimizeMolecule(mol)
                
        #         mol_sdf = Chem.MolToMolBlock(mol)
        #         result.update({f'3d_mol': mol_sdf})
        #         mol_fp = fp.generate_fingerprint_from_molecule(mol, PROTON_FEATURES, scaling='matrix')

        #     # The molecule is a Pybel molecule (Open Babel)
        #     elif isinstance(mol, pb.Molecule):
        #         check_timeout()  # Check timeout before starting operations
        #         result = {"smiles": mol}
        #         try:
        #             result.update({f"smiles_ph": mol.write("smi")})
        #         except Exception as e:
        #             logger.error(f"Error converting pybel molecule to smiles: {e}")
        #             result.update({f"smiles_ph": "N/A"})
        #         result.update({f'3d_mol': mol.write("sdf")})
        #         mol_array = get_array_from_pybelmol(mol)
        #         check_timeout()  # Check timeout
        #         mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')

        #     # The molecule is a CCDC molecule
        #     elif isinstance(mol, ccdcMolecule):
        #         check_timeout()  # Check timeout before operation
        #         mol_array = get_array_from_ccdcmol(mol)
        #         check_timeout()  # Check timeout after generating array
        #         mol_fp = fp.generate_fingerprint_from_data(mol_array, scaling='matrix')

        #     # The molecule is a numpy array
        #     elif isinstance(mol, np.ndarray):
        #         check_timeout()  # Check timeout before processing
        #         mol_fp = fp.generate_fingerprint_from_data(mol, scaling='matrix')

        #     # ---- Calculate the similarity score ----
        #     check_timeout()  # Final check before calculating similarity
        #     sim_score = sim.compute_similarity_score(self.ref_mol_fp, mol_fp)
        #     if np.isnan(sim_score):
        #         sim_score = 0.0
        #     result.update({f'{self.prefix}_HSR_score': sim_score})

        # except TimeoutError:
        #     logger.error(f"Timeout occurred for molecule: {mol}. Processing terminated.")
        #     result.update({
        #         "smiles": mol,
        #         f'3d_mol': 0.0,
        #         f'{self.prefix}_HSR_score': 0.0,
        #         'time_taken': timeout
        #     })

        # except Exception as e:
        #     logger.error(f"Error in score_mol for molecule {mol}: {e}")
        #     result.update({
        #         "smiles": mol,
        #         f'3d_mol': 0.0,
        #         f'{self.prefix}_HSR_score': 0.0,
        #         'error': str(e)
        #     })

        # finally:
        #     # Add the time taken to the result
        #     time_taken = time.time() - start_time
        #     result.update({'time_taken': time_taken})

        # return result


# with Pool(n_processes) as pool:
#             try:
#                 # Use imap to get results in order
#                 for i,result in enumerate(pool.imap(pfunc, molecules)):
#                     results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing molecules: {e}")
#                 results.append({
#                         "smiles": molecules[i],
#                         f'3d_mol': 0.0,
#                         f'{self.prefix}_HSR_score': 0.0,
#                         'time_taken': 'error'
#                     })

# with Pool(n_processes) as pool:

#         # Submit tasks with apply_async
#         async_results = []
#         for mol in molecules:
#             async_result = pool.apply_async(pfunc, args=(mol,))
#             async_results.append(async_result)

#         # Collect results, no longer needing a timeout here as `score_mol` manages it internally
#         for i, async_result in enumerate(async_results):
#             try:
#                 # Get the result without external timeout handling since score_mol has its own
#                 result = async_result.get() 
#             except Exception as e:
#                 # Handle any unexpected exception gracefully
#                 logger.error(f"Error processing molecule {molecules[i]}: {e}")
#                 result = {
#                     "smiles": molecules[i],
#                     f'3d_mol': 0.0,
#                     f'{self.prefix}_HSR_score': 0.0,
#                     'time_taken': 'error'
#                 }
#             finally:
#                 results.append(result)


# for mol in molecules:
#             p = multiprocessing.Process(target=self.score_mol_worker, args=(mol, results_queue))
#             processes.append(p)
#             p.start()

#             # Limit the number of concurrent processes
#             if len(processes) >= n_processes:
#                 # Create a new list to store unfinished processes
#                 remaining_processes = []
#                 for p in processes:
#                     p.join(timeout)
#                     if not p.is_alive():
#                         # Process finished within the timeout; collect the result
#                         try:
#                             result = results_queue.get_nowait()
#                             results.append(result)
#                         except Empty:
#                             # If no result is in the queue, append a default result
#                             logger.warning(f"Process {p.pid} finished but queue was empty.")
#                             results.append({
#                                 f"molecule": mol,
#                                 f'3d_mol': 0.0,
#                                 f'{self.prefix}_HSR_score': 0.0,
#                             })
#                         logger.info(f"Process {p.pid} completed and collected.")
#                     else:
#                         # If the process is still running and exceeded the timeout, terminate it
#                         logger.error(f"Timeout occurred for process PID: {p.pid}, attempting to terminate...")
#                         os.kill(p.pid, signal.SIGKILL)
#                         p.join()  # Ensure the process is cleaned up
#                         logger.info(f"Process {p.pid} terminated.")
#                         results.append({
#                             f"molecule": mol,
#                             f'3d_mol': 0.0,
#                             f'{self.prefix}_HSR_score': 0.0,
#                         })
#                         continue 

#                     # Append the process to the remaining list if it is still active
#                     if p.is_alive():
#                         remaining_processes.append(p)

#                 # Update the list of processes to those still running
#                 processes = remaining_processes

#         # Ensure all remaining processes are joined and results are collected
#         remaining_processes = []
#         for p in processes:
#             p.join(timeout)
#             if not p.is_alive():
#                 # Process finished within the timeout; collect the result
#                 try:
#                     result = results_queue.get_nowait()
#                     results.append(result)
#                 except Empty:
#                     results.append({
#                         f"molecule": mol,
#                         f'3d_mol': 0.0,
#                         f'{self.prefix}_HSR_score': 0.0,
#                     })
#                 logger.info(f"Process {p.pid} completed.")
#             else:
#                 # p.terminate()
#                 os.kill(p.pid, signal.SIGKILL)
#                 p.join()
#                 logger.error(f"Timeout occurred for process PID: {p.pid}, terminated.")
#                 results.append({
#                     f"molecule": mol,
#                     f'3d_mol': 0.0,
#                     f'{self.prefix}_HSR_score': 0.0,
#                 })
            
                
                
    #             def score(self, molecules, directory, file_names):
    #     """
    #     Calculate the scores for a list of molecules in parallel
    #     """
    #     results = []

    #     def process_molecule(mol):
    #         start_time = time.time()
    #         try:
    #             # Timeout handling is achieved by wrapping with try-except
    #             result = self.score_mol(mol)
    #             result['time_taken'] = time.time() - start_time
    #             return result
    #         except Exception as e:
    #             logger.error(f"Timeout/Error occurred for molecule {mol}: {e}")
    #             return {
    #                 "molecule": mol,
    #                 "3d_mol": 0.0,
    #                 f'{self.prefix}_HSR_score': 0.0,
    #                 'time_taken': 'timeout'
    #             }

    #     try:
    #         results = Parallel(n_jobs=self.n_jobs, timeout=self.timeout)(
    #             delayed(process_molecule)(mol) for mol in molecules
    #         )
    #     except Exception as e:
    #         logger.error(f"Error occurred during parallel execution: {e}")

    #     # Save molecules to files as done previously
    #     self.save_results(results, directory, file_names)

    #     return results

    # def save_results(self, results, directory, file_names):
    #     """
    #     Save the results in the specified directory
    #     """
    #     successes = 0
    #     for r, name in zip(results, file_names):
    #         file_path = os.path.join(directory, name + ".sdf")
    #         try:
    #             mol_sdf = r.get('3d_mol')
    #             score = r.get(f"{self.prefix}_HSR_score")
    #             if score > 0.0:
    #                 successes += 1
    #             if mol_sdf:
    #                 with open(file_path, "w") as f:
    #                     f.write(mol_sdf)
    #                     f.write(f"\n> <SMILES>\n{r['molecule']}\n\n$$$$\n")
    #                     f.write(f"\n> <HSR_score>\n{score}\n\n$$$$\n")
    #         except KeyError:
    #             continue
    
    
    # def score(self, molecules: list, directory, file_names, **kwargs):
    #     """
    #     Calculate the scores based on HSR similarity to reference molecules.
    #     :param molecules: List of molecules to score, can be SMILES strings, rdkit molecules or numpy arrays
    #     :param directory: Directory to save files and logs into
    #     :param file_names: List of corresponding file names for SMILES to match files to index
    #     :param kwargs: Ignored
    #     :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
    #     """
    #     # Create directory
    #     step = file_names[0].split("_")[0]  # Assume first Prefix is step
    #     directory = os.path.join(
    #         os.path.abspath(directory), f"{self.prefix}_HSR", step
    #     )
    #     os.makedirs(directory, exist_ok=True)
        
    #     n_processes = min(self.n_jobs, len(molecules), os.cpu_count()-1)
    #     results = []
    #     result_queue = multiprocessing.Queue()
    #     running_processes = []
        
    #     # create a list of tuples with the molecule and the process associated with it
    #     all_processes = [(mol, multiprocessing.Process(target=self.score_mol_worker, args=(mol, result_queue))) for mol in molecules]

    #     for _ in range(n_processes):
    #         if all_processes:
    #             mol, p = all_processes.pop(0)
    #             p.start()
    #             running_processes.append((mol, p, time.time()))
                    
    #     # Polling loop to manage the running processes
    #     while running_processes:
    #         # Create a new list for remaining processes
    #         remaining_processes = []
    #         for mol, p, start_time in running_processes:
    #             # If the process is done, get the result and remove the process from the list
    #             p.join(timeout=0.1)
    #             if not p.is_alive():
    #                 try:
    #                 # Attempt to get the result without blocking
    #                     result = result_queue.get_nowait()
    #                 except Empty:
    #                     # If the result queue is empty, create a default result
    #                     logger.warning(f"Process {p.pid} finished but result queue was empty.")
    #                     result = {
    #                         f"molecule": mol,
    #                         f'3d_mol': 0.0,
    #                         f'{self.prefix}_HSR_score': 0.0,
    #                     }
    #                 results.append(result)
                
    #             # If the process is still running and exceeded the timeout, terminate it
    #             elif time.time() - start_time > self.timeout:
    #                 p.kill()
    #                 p.join()
    #                 results.append({
    #                     f"molecule": mol,
    #                     f'3d_mol': 0.0,
    #                     f'{self.prefix}_HSR_score': 0.0,
    #                 })
    #             # If the process is still running and hasn't timed out, add it back to the list
    #             else:
    #                 remaining_processes.append((mol, p, start_time))

    #         # Update the list of running processes
    #         running_processes = remaining_processes
            
    #         # If there are still processes to be started, start them
    #         while all_processes and len(running_processes) < n_processes:
    #             mol, p = all_processes.pop(0)
    #             p.start()
    #             running_processes.append((mol, p, time.time()))
              
    #     # Save mols
    #     successes = 0
    #     for r, name in zip(results, file_names):
    #         file_path = os.path.join(directory, name + ".sdf")
    #         try:
    #             mol_sdf = r[f"3d_mol"]
    #             score = r[f"{self.prefix}_HSR_score"]
    #             if score > 0.0:
    #                 successes += 1
    #             if mol_sdf:
    #                 with open(file_path, "w") as f:
    #                     f.write(mol_sdf)
    #                     f.write(f"\n> <SMILES>\n{r['smiles']}\n\n$$$$\n")
    #                     f.write(f"\n> <HSR_score>\n{score}\n\n$$$$\n")
                        
    #         except KeyError:
    #             continue
    #     return results
    
    # # Prepare all processes with associated molecules
    #     all_processes = [(mol, multiprocessing.Process(target=self.score_mol_worker, args=(mol, result_queue)))
    #                      for mol in molecules]

    #     # Start a subset of processes up to the number of available cores
    #     for _ in range(n_processes):
    #         if all_processes:
    #             mol, p = all_processes.pop(0)
    #             p.start()
    #             running_processes.append((mol, p, time.time()))

    #     # Polling loop to manage running processes
    #     while running_processes or all_processes:
    #         remaining_processes = []

    #         for mol, p, start_time in running_processes:
    #             p.join(timeout=0.1)  # Allow the process to join with a short timeout

    #             if p.exitcode is not None:  # Process has finished
    #                 if p.exitcode == 0:
    #                     try:
    #                         # Attempt to get the result without blocking
    #                         result = result_queue.get_nowait()
    #                     except Empty:
    #                         # If the result queue is empty, create a default result
    #                         logger.warning(f"Process {p.pid} finished but result queue was empty.")
    #                         result = {
    #                             "molecule": mol,
    #                             '3d_mol': 0.0,
    #                             f'{self.prefix}_HSR_score': 0.0,
    #                         }
    #                 else:
    #                     # Non-zero exit code implies an error occurred
    #                     logger.error(f"Process {p.pid} terminated with exit code {p.exitcode}.")
    #                     result = {
    #                         "molecule": mol,
    #                         '3d_mol': 0.0,
    #                         f'{self.prefix}_HSR_score': 0.0,
    #                     }

    #                 results.append(result)
    #                 p.join()  # Ensure the process resources are cleaned up
    #             elif time.time() - start_time > self.timeout:
    #                 # If the process is still running and exceeded the timeout, terminate it
    #                 logger.error(f"Timeout occurred for process PID: {p.pid}, terminating...")
    #                 os.kill(p.pid, signal.SIGKILL)
    #                 p.join()  
    #                 results.append({
    #                     "molecule": mol,
    #                     '3d_mol': 0.0,
    #                     f'{self.prefix}_HSR_score': 0.0,
    #                 })
    #             else:
    #                 # If the process is still running and hasn't timed out, keep it in the list
    #                 remaining_processes.append((mol, p, start_time))

    #         # Update the list of running processes
    #         running_processes = remaining_processes

    #         # If there are still processes to be started, start them
    #         while all_processes and len(running_processes) < n_processes:
    #             mol, p = all_processes.pop(0)
    #             p.start()
    #             running_processes.append((mol, p, time.time()))

    #     return results