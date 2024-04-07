import os
import argparse
import logging
import numpy as np
from flask import Flask, request, jsonify

import chemprop

logger = logging.getLogger("chemprop_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = Flask(__name__)


class Model:
    """
    This particular class uses the QSAR model from Lib-INVENT in a stand alone environment to avoid conflict dependencies.
    """

    def __init__(self, model_dir: os.PathLike, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (specifically clf.pkl)
        """
        predict_arguments = [
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
            "--checkpoint_dir",
            model_dir,
        ]
        self.predict_args = chemprop.args.PredictArgs().parse_args(predict_arguments)

        logger.debug(f"Loading model from {model_dir}")
        self.model_objects = chemprop.train.load_model(args=self.predict_args)
        logger.debug(f"Model loaded: {self.model_objects}")


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    logger.debug("POST request received")
    smiles = request.get_json().get("smiles", [])
    logger.debug(f"Reading SMILES:\n\t{smiles}")

    # Make predictions
    # Preds & uncs are -> {idx: pred, idx2: pred2}
    # while individual pred/uncd is shape(tasks)
    preds, uncs = chemprop.train.make_predictions(
        args=model.predict_args,
        smiles=[[smi] for smi in smiles],
        model_objects=model.model_objects,
        return_invalid_smiles=True,
        return_index_dict=True,
        return_uncertainty=True,
    )
    # Returns format (molidx: [avg_endpoint1, avg_endpoint2, ...]) If invalid, returens ['Invalid SMILES']
    # Preds = {0: [3.3227710232816094], 1: [2.9169101045436707], 2: [3.6473884211186487], 3: [1.9292660002520876], 4: [3.1925854582522613]}
    # Uncs = {0: array([nan]), 1: array([nan]), 2: array([nan]), 3: array([nan]), 4: array([nan])}
    results = []
    for i, smi in enumerate(smiles):
        r = {"smiles": smi}
        r.update(
            {
                f"pred{pi+1}": 0.0 if p == "Invalid SMILES" else p
                for pi, p in enumerate(preds[i])
            }
        )
        r.update(
            {
                "mean_pred": np.mean([p for p in preds[i] if p != "Invalid SMILES"]),
                "max_pred": np.max([p for p in preds[i] if p != "Invalid SMILES"]),
                "min_pred": np.min([p for p in preds[i] if p != "Invalid SMILES"]),
            }
        )
        r.update(
            {
                f"unc{ui+1}": 0.0 if u == "Invalid SMILES" else u
                for ui, u in enumerate(uncs[i])
            }
        )
        r.update(
            {
                "mean_unc": np.mean([u for u in uncs[i] if u != "Invalid SMILES"]),
                "max_unc": np.max([u for u in uncs[i] if u != "Invalid SMILES"]),
                "min_unc": np.min([u for u in uncs[i] if u != "Invalid SMILES"]),
            }
        )
        results.append(r)

    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument(
        "--model_dir", type=str, help="Path to pre-trained model (e.g., clf.pkl)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port)
