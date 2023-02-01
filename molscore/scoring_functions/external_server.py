import os
import requests
import argparse
import logging
import numpy as np
from functools import partial
from tempfile import NamedTemporaryFile

from molscore.scoring_functions.utils import get_mol, Fingerprints, SimilarityMeasures
from multiprocessing import Pool
from rdkit.Chem import AllChem as Chem
from typing import Union

from molscore.scoring_functions.utils import get_mol

logger = logging.getLogger('external_server')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class POSTServer:
    """ POST generated molecules to an external server based scoring function, assumes they are returned in the same order! """
    return_metrics = ['server_value']

    def __init__(self, prefix, server_address: str, input_format: str, output_value: str):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param server_address: Server address used to run a scoring function
        :param input_format: How to prepare data to be sent to server [smi, sdf]
        :param output_value: The name of output metric to be used (assumes JSON output format of a list of dictionaries) 
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        self.server_address = server_address
        self.input_format = input_format
        self.output_value = output_value

    def prepare_input(self, smiles, names, file):
        """ Write input file in specified format. """
        if self.input_format == 'smi':
            for smi in smiles:
                file.write(f'{smi}\n')
        elif self.input_format == 'sdf':
            writer = Chem.SDWriter(file)
            for smi, name in zip(smiles, names):
                mol = get_mol(smi)
                mol.SetProp('_Name', name)
                writer.write(mol)
            writer.close()
        else:
            raise ValueError(f"Unrecognized input format: {self.input_format}")
        return 

    def score(self, smiles: list, directory: str, file_names: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [{'smiles': smi} for smi in smiles]
        # Only submit valid smiles
        valid_smiles = [i for i, smi in enumerate(smiles) if get_mol(smi)]
        # Use tempfile
        with NamedTemporaryFile(mode='w+t', suffix=f".{self.input_format}") as tfile:
            self.prepare_input(
                smiles=[smiles[i] for i in valid_smiles],
                names=[file_names[i] for i in valid_smiles],
                file=tfile
                )
            # Submit
            res = requests.post(self.server_address, data={"sdf": tfile.name})
            data = res.json()['data']
            all_keys = set()  # Keep record of keys
            for i, out in zip(valid_smiles, data):  # Ensure input and output are equal length
                for k in out:
                    all_keys.add(k)
                    value = out[k] if out[k] else 0.0
                    results[i].update({f'{self.prefix}_{k}': value})
                    # Record named value twice as this is the name used
                    if k == self.output_value:
                        results[i].update({f'{self.prefix}_{self.output_value}': value})

        # Add keys as 0.0 for invalid molecules
        for i, res in enumerate(results):
            if i not in valid_smiles:
                res.update({f'{self.prefix}_{k}': 0.0 for k in all_keys})
        
        return res
    
    def __call__(self, smiles: list, directory: str, file_names: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        logger.warning("__call__() will be deprecated in future versions, please use score() instead.")
        return self.score(smiles=smiles, directory=directory, file_names=file_names)
