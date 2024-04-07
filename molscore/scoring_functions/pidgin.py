import gzip
import json
import logging
import pickle as pkl
import zipfile
from functools import partial

import numpy as np

from molscore.scoring_functions.utils import Fingerprints, Pool, Zenodo

logger = logging.getLogger("pidgin")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class PIDGIN:
    """
    Download and run PIDGIN classification models (~11GB) via Zenodo to return the positive predictions, atleast one uniprot must be specified.
    """

    return_metrics = ["pred_proba"]

    zenodo = Zenodo(
        access_token="0"
    )  # Set access token to zero as it's required zenodo-client API but not Zenodo for downloading public data

    @classmethod
    def get_pidgin_record_id(cls):
        # This feature is broken in zenodo-client currently (17/10/23, v0.3.2): returns 'latest' and not actual record id
        return "7547691"

    # Download list of uniprots
    @classmethod
    def get_uniprot_list(cls):
        uniprots_path = cls.zenodo.download(
            record_id=cls.get_pidgin_record_id(), name="uniprots.json"
        )
        with open(uniprots_path, "rt") as f:
            uniprots = json.load(f)
        uniprots = ["None"] + uniprots
        return uniprots

    # Download uniprot groups
    @classmethod
    def get_uniprot_groups(cls):
        groups = {"None": None}
        groups_path = cls.zenodo.download(
            record_id=cls.get_pidgin_record_id(), name="uniprots_groups.json"
        )
        with open(groups_path, "rt") as f:
            groups.update(json.load(f))
        return groups

    # Set init docstring here as it's not a string literal
    @classmethod
    def set_docstring(cls):
        init_docstring = f"""
            :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
            :param uniprot: Uniprot accession for classifier to use [{', '.join(cls.get_uniprot_list())}]
            :param uniprots: List of uniprot accessions for classifier to use
            :param uniprot_set: Set of uniprots based on protein class (level - name - size) [{', '.join(cls.get_uniprot_groups().keys())}]
            :param exclude_uniprot: Uniprot to exclude (useful to remove from a uniprot set) [{', '.join(cls.get_uniprot_list())}]
            :param exclude_uniprots: Uniprot list to exclude (useful to remove from a uniprot set)
            :param thresh: Concentration threshold of classifier [100 uM, 10 uM, 1 uM, 0.1 uM]
            :param method: How to aggregate the positive prediction probabilities accross classifiers [mean, median, max, min]
            :param binarise: Binarise predicted probability and return ratio of actives based on optimal predictive thresholds (GHOST)
            :param kwargs:
            """
        setattr(cls.__init__, "__doc__", init_docstring)

    def __init__(
        self,
        prefix: str,
        uniprot: str = None,
        uniprots: list = None,
        uniprot_set: str = None,
        thresh: str = "100 uM",
        exclude_uniprot: str = None,
        exclude_uniprots: list = None,
        n_jobs: int = 1,
        method: str = "mean",
        binarise=False,
        **kwargs,
    ):
        """This docstring is must be populated by calling PIDGIN.set_docstring() first."""
        # Make sure something is selected
        self.uniprot = uniprot if uniprot != "None" else None
        self.uniprots = uniprots if uniprots is not None else []
        self.uniprot_set = uniprot_set if uniprot_set != "None" else None
        self.exclude_uniprot = exclude_uniprot if exclude_uniprot != "None" else None
        self.exclude_uniprots = exclude_uniprots if exclude_uniprots is not None else []
        assert (
            (self.uniprot is not None)
            or (len(self.uniprots) > 0)
            or (self.uniprot_set is not None)
        ), "Either uniprot, uniprots or uniprot set must be specified"
        # Set other attributes
        self.prefix = prefix.replace(" ", "_")
        self.thresh = thresh.replace(" ", "").replace(".", "")
        self.n_jobs = n_jobs
        self.models = []
        self.model_names = []
        self.ghost_thresholds = []
        self.fp = "ECFP4"
        self.nBits = 2048
        self.agg = getattr(np, method)
        self.binarise = binarise
        # Curate uniprot set
        self.pidgin_record_id = self.get_pidgin_record_id()
        self.groups = self.get_uniprot_groups()
        if self.uniprot:
            self.uniprots += [self.uniprot]
        if self.uniprot_set:
            self.uniprots += self.groups[self.uniprot_set]
        if self.exclude_uniprot:
            self.exclude_uniprots += [self.exclude_uniprot]
        for uni in self.exclude_uniprots:
            if uni in self.uniprots:
                self.uniprots.remove(uni)
        # De-duplicate
        self.uniprots = list(set(self.uniprots))

        # Download PIDGIN
        logger.warning(
            "If not downloaded, PIDGIN will be downloaded which is a large file ~ 11GB and may take some several minutes"
        )
        pidgin_path = self.zenodo.download(
            record_id=self.pidgin_record_id, name="trained_models.zip"
        )
        with zipfile.ZipFile(pidgin_path, "r") as zip_file:
            for uni in self.uniprots:
                try:
                    # Load .json to get ghost thresh
                    with zip_file.open(f"{uni}.json") as meta_file:
                        metadata = json.load(meta_file)
                        opt_thresh = metadata[thresh]["train"]["params"][
                            "opt_threshold"
                        ]
                        self.ghost_thresholds.append(opt_thresh)
                    # Load classifier
                    with zip_file.open(f"{uni}_{self.thresh}.pkl.gz") as model_file:
                        with gzip.open(model_file, "rb") as f:
                            clf = pkl.load(f)
                            self.models.append(clf)
                            self.model_names.append(f"{uni}@{self.thresh}")
                except (FileNotFoundError, KeyError):
                    logger.warning(f"{uni} model at {thresh} not found, omitting")
                    continue

        # Run some checks
        assert len(self.models) != 0, "No models were found"
        if self.binarise:
            logger.info("Running with binarise=True so setting method=mean")
            self.agg = np.mean
            assert len(self.ghost_thresholds) == len(
                self.models
            ), "Mismatch between models and thresholds"

    def score(self, smiles: list, **kwargs):
        """
        Calculate scores for an sklearn model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [{"smiles": smi, f"{self.prefix}_pred_proba": 0.0} for smi in smiles]
        valid = []
        fps = []
        predictions = []
        aggregated_predictions = []
        # Calculate fps
        with Pool(self.n_jobs) as pool:
            pcalculate_fp = partial(
                Fingerprints.get, name=self.fp, nBits=self.nBits, asarray=True
            )
            [
                (valid.append(i), fps.append(fp))
                for i, fp in enumerate(pool.imap(pcalculate_fp, smiles))
                if fp is not None
            ]

        # Return early if there's no results
        if len(fps) == 0:
            for r in results:
                r.update({f"{self.prefix}_{name}": 0.0 for name in self.model_names})
                r.update({f"{self.prefix}_pred_proba": 0.0})
            return results

        # Predict
        for clf in self.models:
            prediction = clf.predict_proba(np.asarray(fps).reshape(len(fps), -1))[
                :, 1
            ]  # Input (smiles, bits)
            predictions.append(prediction)
        predictions = np.asarray(predictions).transpose()  # Reshape to (smiles, models)

        # Binarise
        if self.binarise:
            thresh = np.asarray(self.ghost_thresholds)
            predictions = (predictions >= thresh).astype(int)

        # Update results
        for i, row in zip(valid, predictions):
            for j, name in enumerate(self.model_names):
                results[i].update(
                    {f"{self.prefix}_{name}": float(row[j])}
                )  # JSON doesn't like numpy types

        # Aggregate
        aggregated_predictions = self.agg(predictions, axis=1)

        # Update results
        for i, prob in zip(valid, aggregated_predictions):
            results[i].update(
                {f"{self.prefix}_pred_proba": float(prob)}
            )  # JSON doesn't like numpy types

        return results

    def __call__(self, smiles: list, **kwargs):
        return self.score(smiles=smiles)
