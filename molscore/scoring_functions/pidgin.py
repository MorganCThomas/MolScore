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
    uniprots = ['None'] + uniprots

    # Download uniprot groups
    uniprot_groups = {'None': None}
    groups_path = zenodo.download_latest(record_id=pidgin_record_id, path='uniprots_groups.json')
    with open(groups_path, 'rt') as f:
        uniprot_groups.update(json.load(f))

    # Set init docstring here as it's not a string literal
    init_docstring = f"""
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param uniprot: Uniprot accession for classifier to use [{', '.join(uniprots)}]
        :param uniprots: List of uniprot accessions for classifier to use
        :param uniprot_set: Set of uniprots based on protein class (level - name - size) [{', '.join(uniprot_groups.keys())}]
        :param thresh: Concentration threshold of classifier [100 uM, 10 uM, 1 uM, 0.1 uM]
        :param method: How to aggregate the positive prediction probabilities accross classifiers [mean, median, max, min]
        :param binarise: Binarise predicted probability and return ratio of actives based on optimal predictive thresholds (GHOST)
        :param kwargs:
        """

    def __init__(
        self, prefix: str, uniprot: str = None, uniprots: list = None, uniprot_set: str = None, thresh: str = '100 uM',
        n_jobs: int = 1, method: str = 'mean', binarise=False, **kwargs):
        """This docstring is replaced outside this method."""
        # Make sure something is selected
        self.uniprot = uniprot if uniprot != 'None' else None
        self.uniprots = uniprots if uniprots is not None else []
        self.uniprot_set = uniprot_set if uniprot_set != 'None' else None
        assert (self.uniprot is not None) or (len(self.uniprots) > 0) or (self.uniprot_set is not None), "Either uniprot, uniprots or uniprot set must be specified"
        # Set other attributes
        self.prefix = prefix.replace(" ", "_")
        self.thresh = thresh.replace(" ", "").replace(".", "")
        self.n_jobs = n_jobs
        self.models = []
        self.ghost_thresholds = []
        self.fp = 'ECFP4'
        self.nBits = 2048
        self.agg = getattr(np, method)
        self.binarise = binarise
        # Curate uniprot set
        if self.uniprot:
            self.uniprots += self.uniprot
        if self.uniprot_set:
            self.uniprots += self.uniprot_groups[self.uniprot_set]
        # De-duplicate
        self.uniprots = list(set(self.uniprots))

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

        # Run some checks
        assert len(self.models) != 0, "No models were found"
        if self.binarise:
            logger.info('Running with binarise=True so setting method=mean')
            self.agg = np.mean
            assert len(self.ghost_thresholds) == len(self.models), "Mismatch between models and thresholds"

    setattr(__init__, '__doc__', init_docstring)

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

    

    


