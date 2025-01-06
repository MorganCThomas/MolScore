import glob
import os
import pickle as pkl
from functools import partial
from typing import Union

import joblib
import numpy as np
from rdkit import rdBase

from molscore.scoring_functions.utils import Fingerprints, Pool

rdBase.DisableLog("rdApp.error")


class SKLearnClassifier:
    """
    Score structures by loading a pre-trained sklearn classifier and return the predicted values
    """

    return_metrics = ["pred_proba"]

    def __init__(
        self,
        prefix: str,
        model_path: Union[str, os.PathLike],
        fp: str,
        nBits: int = 1024,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (saved using joblib)
        :param fp: What type of fingerprint to use [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, hashAP, hashTT, RDK5, RDK6, RDK7]
        :param nBits: Length of fingerprint
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        self.prefix = prefix
        self.prefix = prefix.replace(" ", "_")
        self.model_path = model_path
        self.fp = fp
        self.nBits = int(nBits)
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs)

        # Load in model and assign to attribute
        if model_path.endswith(".joblib"):
            self.model = joblib.load(model_path)
        elif model_path.endswith(".pkl") or model_path.endswith(".pickle"):
            with open(model_path, "rb") as f:
                self.model = pkl.load(f)
        else:
            raise TypeError(f"Unrecognized file extension: {os.rsplit('.', 1)[-1]}")

    def __call__(self, smiles: list, **kwargs):
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
        pcalculate_fp = partial(
            Fingerprints.get, name=self.fp, nBits=self.nBits, asarray=True
        )
        [
            (valid.append(i), fps.append(fp.reshape(1, -1)))
            for i, fp in enumerate(self.mapper(pcalculate_fp, smiles))
            if fp is not None
        ]

        if len(valid) != 0:
            probs = self.model.predict_proba(np.asarray(fps).reshape(len(fps), -1))[
                :, 1
            ]
            for i, prob in zip(valid, probs):
                results[i].update({f"{self.prefix}_pred_proba": prob})

        return results


class SKLearnRegressor(SKLearnClassifier):
    """
    Score structures by loading a pre-trained sklearn regressor and return the predicted values
    """

    return_metrics = ["predict"]

    def __init__(
        self,
        prefix: str,
        model_path: os.PathLike,
        fp: str,
        nBits: int = 1024,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (saved using joblib)
        :param fp: What type of fingerprint to use [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, hashAP, hashTT, RDK5, RDK6, RDK7]
        :param nBits: Length of fingerprint
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        super().__init__(
            prefix=prefix, model_path=model_path, fp=fp, nBits=nBits, n_jobs=n_jobs
        )

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for an sklearn model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """

        results = [{"smiles": smi, f"{self.prefix}_predict": 0.0} for smi in smiles]
        valid = []
        fps = []
        pcalculate_fp = partial(
            Fingerprints.get, name=self.fp, nBits=self.nBits, asarray=True
        )
        [
            (valid.append(i), fps.append(fp.reshape(1, -1)))
            for i, fp in enumerate(self.mapper(pcalculate_fp, smiles))
            if fp is not None
        ]

        if len(valid) != 0:
            preds = self.model.predict(np.asarray(fps).reshape(len(fps), -1))
            for i, pred in zip(valid, preds):
                results[i].update({f"{self.prefix}_predict": pred})

        return results


# Backwards compatability
SKLearnModel = SKLearnClassifier


class EnsembleSKLearnModel(SKLearnModel):
    """
    This class loads different random seeds of a defined sklearn
    model and returns the average of the predicted values.
    """

    def __init__(
        self,
        prefix: str,
        model_path: os.PathLike,
        fp: str,
        nBits: int,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (saved using joblib)
        :param fp: What type of fingerprint to use [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, hashAP, hashTT, RDK5, RDK6, RDK7]
        :param nBits: Length of fingerprint
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        super().__init__(prefix, model_path, fp, nBits, n_jobs, **kwargs)
        changing = self.model_path.split("_")
        del changing[len(changing) - 1]
        changing = "_".join(changing)
        self.replicates = sorted(glob.glob(str(changing) + "*.joblib"))

        # Load in model and assign to attribute
        self.models = []
        for f in self.replicates:
            self.models.append(joblib.load(f))

    def __call__(self, smiles: list, **kwargs):
        results = [{"smiles": smi, f"{self.prefix}_pred_proba": 0.0} for smi in smiles]
        valid = []
        fps = []
        predictions = []
        averages = []
        with Pool(self.n_jobs) as pool:
            pcalculate_fp = partial(
                Fingerprints.get, name=self.fp, nBits=self.nBits, asarray=True
            )
            [
                (valid.append(i), fps.append(fp))
                for i, fp in enumerate(pool.imap(pcalculate_fp, smiles))
                if fp is not None
            ]

        # Predicting the probabilies and appending them to a list.
        for m in self.models:
            prediction = m.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
            predictions.append(prediction)
        predictions = np.asarray(predictions)
        averages = predictions.mean(axis=0)

        for i, prob in zip(valid, averages):
            results[i].update({f"{self.prefix}_pred_proba": prob})

        return results
