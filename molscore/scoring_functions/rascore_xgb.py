"""
Adapted from https://github.com/reymond-group/RAscore published https://doi.org/10.1039/d0sc05401a
"""
import os
import tempfile
import logging
import subprocess
import pandas as pd
import numpy as np
import pickle as pkl
from importlib import resources


from rdkit import Chem
from rdkit.Chem import AllChem

from molscore.scoring_functions.utils import get_mol, timedSubprocess, Pool

logger = logging.getLogger('rascore_xgb')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class RAScore_XGB:
    """
    Predicted synthetic feasibility according to solveability by AiZynthFinder https://doi.org/10.1039/d0sc05401a
    """
    return_metrics = ['pred_proba']

    def __init__(self, prefix: str = 'RAScore', model: str = 'ChEMBL', method: str = 'XGB', **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance
        :param model: Either ChEMBL, GDB, GDBMedChem [ChEMBL, GDB, GDBMedChem]
        :param method: Either XGB or DNN [XGB, DNN]
        :param kwargs:
        """
        self.prefix = prefix
        self.subprocess = timedSubprocess()
        self.env = "rascore-env"
        self.method = method
        if self.method == 'XGB':
            self.ext = 'pkl'
            self.fp = 'ecfp'
        else:
            self.ext = 'h5'
            self.fp = 'fcfp'
        
        # Check/create RAscore Environment
        if not self._check_env():
            logger.warning(f"Failed to identify {self.env}, attempting to create it automatically (this may take several minutes)")
            self._create_env()
            logger.info(f"{self.env} successfully created")
        else:
            logger.info(f"Found existing {self.env}")

        if model == 'ChEMBL':
            with resources.path(f'molscore.data.models.RAScore.{self.method}_chembl_{self.fp}_counts', f'model.{self.ext}') as p:
                self.model_path = str(p)
        elif model == 'GDB':
            with resources.path(f'molscore.data.models.RAScore.{self.method}_gdbchembl_{self.fp}_counts', f'model.{self.ext}') as p:
                self.model_path = str(p)
        elif model == 'GDBMedChem':
            with resources.path(f'molscore.data.models.RAScore.{self.method}_gdbmedechem_{self.fp}_counts', f'model.{self.ext}') as p:
                self.model_path = str(p)      
        else:
            raise "Please select from ChEMBL, GDB or GDBMedChem"


    def _check_env(self):
        cmd = "conda info --envs"
        out = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        envs = [line.split(" ")[0] for line in out.stdout.decode().splitlines()[2:]]
        return self.env in envs

    
    def _create_env(self):
        cmd = f"conda create -n {self.env} python=3.7 -y ; " \
              f"conda install -n {self.env} -c rdkit rdkit -y ; " \
              f"conda run -n {self.env} pip install git+https://github.com/reymond-group/RAscore.git@master"
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create {self.env} automatically please install as per instructions https://github.com/reymond-group/RAscore")
            raise e


    @staticmethod
    def calculate_fp(smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.
        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        raise DeprecationWarning
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False)
            size = 2048
            arr = np.zeros((size,), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                arr[nidx] += int(v)
            return arr
        else:
            return None

    def _score_old(self, smiles: list, **kwargs):
        """
        Calculate RAScore given a list of SMILES, if a smiles is abberant or invalid,
        should return 0.0

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        raise DeprecationWarning
        results = [{'smiles': smi, f'{self.prefix}_pred_proba': 0.0} for smi in smiles]
        valid = []
        fps = []
        with Pool(self.n_jobs) as pool:
            [(valid.append(i), fps.append(fp))
             for i, fp in enumerate(pool.imap(self.calculate_fp, smiles))
             if fp is not None]
        probs = self.model.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
        for i, prob in zip(valid, probs):
            results[i].update({f'{self.prefix}_pred_proba': prob})

        return results

    def score(self, smiles: list, directory: str, **kwargs):
        """
        Calculate RAScore given a list of SMILES, if a smiles is abberant or invalid,
        should return 0.0

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        directory = os.path.abspath(directory)
        # Populate results with 0.0 placeholder
        results = [{'smiles': smi} for smi in smiles]
        for result in results:
            result.update({f"{self.prefix}_{metric}": 0.0 for metric in self.return_metrics})
        # Ensure they're valid otherwise AiZynthFinder will throw an error
        valid = [i for i, smi in enumerate(smiles) if get_mol(smi)]
        if len(valid) == 0:
            return results
        # Write smiles to a tempfile
        smiles_file = tempfile.NamedTemporaryFile(mode='w+t', dir=directory, suffix='.smi')
        smiles_file.write('SMILES\n')
        for i in valid:
            smiles_file.write(smiles[i]+'\n')
        smiles_file.flush()
        # Specify output file
        output_file = os.path.join(directory, 'rascore_out.csv')
        # Submit job to aizynthcli (specify filter policy if not None)
        cmd = f"conda run -n {self.env} " \
              f"RAscore -f {smiles_file.name} -o {output_file} -m {self.model_path}"
        self.subprocess.run(cmd=cmd,cwd=directory)
        # Read in ouput
        output_data = pd.read_csv(output_file)
        # Process output
        for i, score in zip(valid, output_data.RAscore):
            results[i].update({f"{self.prefix}_pred_proba": score})
        return results


    def __call__(self, smiles: list, directory, **kwargs):
        """
        Calculate RAScore given a list of SMILES, if a smiles is abberant or invalid,
        should return 0.0

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles, directory=directory)
