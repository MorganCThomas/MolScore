import argparse
import gzip
import json
import logging
import pickle as pkl
import re
import zipfile
from functools import partial

import numpy as np
from flask import Flask, jsonify, request
from pidgin_utils import Zenodo
from utils import Fingerprints, Pool

logger = logging.getLogger("pidgin_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


app = Flask(__name__)


class Model:
    """
    Create a Model object that loads models once, for example.
    """

    zenodo = Zenodo(access_token="0")
    pidgin_record_id = "7547691"

    def __init__(
        self,
        uniprots: list,
        thresh: str,
        method: str,
        binarise: bool,
        n_jobs: int,
        **kwargs,
    ):
        self.uniprots = uniprots
        self.thresh = thresh
        self.full_thresh = re.sub("01", "0.1", thresh)
        self.full_thresh = thresh[:-2] + " " + thresh[-2:]
        self.binarise = binarise
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.fp = "ECFP4"
        self.nBits = 2048
        self.agg = getattr(np, method)
        self.ghost_thresholds = []
        self.models = []
        self.model_names = []

        # Load models
        pidgin_path = self.zenodo.download(
            record_id=self.pidgin_record_id, name="trained_models.zip"
        )
        with zipfile.ZipFile(pidgin_path, "r") as zip_file:
            for uni in self.uniprots:
                try:
                    # Load .json to get ghost thresh
                    with zip_file.open(f"{uni}.json") as meta_file:
                        metadata = json.load(meta_file)
                        # reformart thresh back
                        opt_thresh = metadata[self.full_thresh]["train"]["params"][
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
        if len(self.models) != 0:
            logger.warning("No models were found")
        if self.binarise:
            logger.info("Running with binarise=True so setting method=mean")
            self.agg = np.mean
            assert len(self.ghost_thresholds) == len(
                self.models
            ), "Mismatch between models and thresholds"


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    smiles = request.get_json().get("smiles", [])

    results = [{"smiles": smi, "pred_proba": 0.0} for smi in smiles]
    valid = []
    fps = []
    predictions = []
    aggregated_predictions = []

    # Calculate fingerprints
    pcalculate_fp = partial(
        Fingerprints.get, name=model.fp, nBits=model.nBits, asarray=True
    )
    [
        (valid.append(i), fps.append(fp))
        for i, fp in enumerate(model.mapper(pcalculate_fp, smiles))
        if fp is not None
    ]

    # Return early if there's no results
    if len(fps) == 0:
        for r in results:
            r.update({"{name}": 0.0 for name in model.model_names})
            r.update({"pred_proba": 0.0})
        return jsonify(results)

    # Predict
    for clf in model.models:
        prediction = clf.predict_proba(np.asarray(fps).reshape(len(fps), -1))[
            :, 1
        ]  # Input (smiles, bits)
        predictions.append(prediction)
    predictions = np.asarray(predictions).transpose()  # Reshape to (smiles, models)

    # Binarise
    if model.binarise:
        thresh = np.asarray(model.ghost_thresholds)
        predictions = (predictions >= thresh).astype(int)

    # Update results
    for i, row in zip(valid, predictions):
        for j, name in enumerate(model.model_names):
            results[i].update({f"{name}": float(row[j])})

    # Aggregate
    try:
        aggregated_predictions = model.agg(predictions, axis=1)
    except np.AxisError:
        # If no models exist and cannot be aggregated
        aggregated_predictions = []

    # Update results
    for i, prob in zip(valid, aggregated_predictions):
        results[i].update({"pred_proba": float(prob)})

    # Return results
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument(
        "--thresh",
        type=str,
        default="100 uM",
        help="Concentration threshold of classifier [100uM, 10uM, 1uM, 01uM]",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean",
        help="How to aggregate the positive prediction probabilities accross classifiers [mean, median, max, min]",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="")
    parser.add_argument("--uniprots", nargs="+", help="Which uniprots to run")
    parser.add_argument(
        "--binarise",
        action="store_true",
        help="Binarise predicted probability and return ratio of actives based on optimal predictive thresholds (GHOST)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port, debug=False)
