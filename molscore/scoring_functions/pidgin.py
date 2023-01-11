import os
import gzip
import json
import zipfile
import logging
import pickle as pkl
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
from typing import Union
from zenodo_client import Zenodo

from molscore.scoring_functions.utils import Fingerprints

logger = logging.getLogger('pidgin')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class PIDGIN:
    """
    Download and run PIDGIN classification models (~11GB) via Zenodo to return the positive predictions
    """
    return_metrics = ['pred_proba']

    zenodo = Zenodo()
    pidgin_record_id = zenodo.get_latest_record('7504135')

    # Download list of uniprots
    uniprots_path = zenodo.download_latest(record_id=pidgin_record_id, path='uniprots.json')
    with open(uniprots_path, 'rt') as f:
        uniprots = json.load(f)
    # Download uniprot sets
    classification_path = zenodo.download_latest(record_id=pidgin_record_id, path='uniprots_classification.csv')
    uni_df = pd.read_csv(classification_path)
    uni_classes = uni_df.groupby(['pref_name', 'class_level'])['accession'].count().reset_index().sort_values(['class_level', 'pref_name'])
    uni_classes = uni_classes.astype(str)[['class_level', 'pref_name', 'accession']].agg(' - '.join, axis=1).tolist()
    # TODO sort this out, class level doesn't include child classes ... uniprot count isn't unique, need to sort original file out


    def __init__(
        self, prefix: str, uniprots: Union[str, list] = None, uniprot_set: str = None, thresh: str = '100 uM',
        n_jobs: int = 1, method: str = 'mean', binarise=False
        ):
        f"""
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param uniprots: Uniprot accession of classifier to use [{', '.join(self.uniprots)}]
        :param uniprot_set: Set of uniprots based on protein class (level - name - size) [{', '.join(self.uni_classes)}]
        :param thresh: Concentration threshold of classifier [100 uM, 10 uM, 1 uM, 0.1 uM]
        :param method: How to aggregate the positive predictions accross classifiers [mean, median, max, min]
        :param binarise: Binarise predicted probability and return ratio of actives based on optimal predictive thresholds (GHOST)
        """
        assert (uniprots is not None) or (uniprot_set is not None), "Either uniprots or uniprot set must be specified"
        self.prefix = prefix.replace(" ", "_")
        self.thresh = thresh.replace(" ", "").replace(".", "")
        self.n_jobs = n_jobs
        self.models = []
        self.ghost_thresholds = []
        self.fp = 'ECFP4'
        self.nBits = 2048
        self.agg = getattr(np, method)
        self.binarise = binarise
        if uniprots:
            if isinstance(uniprots, str):
                self.uniprots = [uniprots]
            else:
                self.uniprots = uniprots

        # Get uniprot set
        if uniprot_set:
            try:
                pref_name = uniprot_set.split(" - ")[1]  # Assume level - name - size
            except ValueError:
                pref_name = uniprot_set.split(" - ")[0]  # Assume name
            # Get associated uniprots
            set_uniprots = list(self.uni_df.loc[self.uni_df.pref_name == pref_name, 'accession'].unique())
            if uniprots:
                # Add to uniprots if also specified
                self.uniprots += set_uniprots
                self.uniprots = list(set(self.uniprots)) # De-duplicate
            else:
                self.uniprots = set_uniprots

        # Download PIDGIN
        logger.warning('If not downloaded, PIDGIN will be downloaded which is a large file ~ 11GB and may take some several minutes')
        pidgin_path = self.zenodo.download_latest(record_id=self.pidgin_record_id, path='trained_models.zip')
        with zipfile.ZipFile(pidgin_path, 'r') as zip_file:
            for uni in self.uniprots:
                try:
                    # Load .json to get ghost thresh
                    with zip_file.open(f'{uni}.json') as meta_file:
                            metadata = json.load(meta_file)
                            opt_thresh = metadata[thresh]['train']['params']['opt_threshold']
                            self.ghost_thresholds.append(opt_thresh)
                    # Load classifier
                    with zip_file.open(f'{uni}_{self.thresh}.pkl.gz') as model_file:
                        with gzip.open(model_file, 'rb') as f:
                            clf = pkl.load(f)
                            self.models.append(clf)
                except (FileNotFoundError, KeyError):
                    logger.warning(f'{uni} model at {thresh} not found, omitting')
                    continue

        # Download models
        if False:
            for uni in self.uniprots:
                # Download it
                try:
                    uni_zip = self.zenodo.download_latest(record_id=self.pidgin_record_id, path=f'{uni.upper()}.zip')
                except FileNotFoundError:
                    logger.warning(f'{uni} models not found, omitting')
                    continue
                # Load it
                try:
                    with zipfile.ZipFile(uni_zip, 'r') as zip_file:
                        # Load .json to get ghost thresh
                        with zip_file.open(f'{uni}.json') as meta_file:
                                metadata = json.load(meta_file)
                                opt_thresh = metadata[thresh]['train']['params']['opt_threshold']
                                self.ghost_thresholds.append(opt_thresh)
                        # Load classifier
                        with zip_file.open(f'{uni}_{self.thresh}.pkl.gz') as model_file:
                            with gzip.open(model_file, 'rb') as f:
                                clf = pkl.load(f)
                                self.models.append(clf)
                except (FileNotFoundError, KeyError):
                    logger.warning(f'{uni} model at {thresh} not found, omitting')
                    continue

        # Run some checks
        assert len(self.models) != 0, "No models were found"
        if self.binarise:
            logger.info('Running with binarise=True so setting method=mean')
            self.agg = np.mean
            assert len(self.ghost_thresholds) == len(self.models), "Mismatch between models and thresholds"

    def score(self, smiles: list, **kwargs):
        """
        Calculate scores for an sklearn model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [{'smiles': smi, f'{self.prefix}_pred_proba': 0.0} for smi in smiles]
        valid = []
        fps = []
        predictions = []
        aggregated_predictions = []
        # Calculate fps
        with Pool(self.n_jobs) as pool:
            pcalculate_fp = partial(Fingerprints.get, name=self.fp, nBits=self.nBits, asarray=True)
            [(valid.append(i), fps.append(fp))
            for i, fp in enumerate(pool.imap(pcalculate_fp, smiles))
            if fp is not None]
        
        # Predict
        for clf in self.models:
            prediction = clf.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
            predictions.append(prediction)
        predictions = np.asarray(predictions)

        # Binarise
        if self.binarise:
            thresh = np.asarray(self.ghost_thresholds).reshape(-1, 1)
            predictions = (predictions >= thresh)

        # Aggregate
        aggregated_predictions = self.agg(predictions, axis=0)

        # Update results
        for i, prob in zip(valid, aggregated_predictions):
            results[i].update({f'{self.prefix}_pred_proba': prob})

        return results

    def __call__(self, smiles: list, **kwargs):
        return self.score(smiles=smiles)

    

    


