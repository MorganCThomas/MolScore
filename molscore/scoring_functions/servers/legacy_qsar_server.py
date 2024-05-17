import argparse
import logging
import os
import pickle as pkl

import numpy as np
from flask import Flask, jsonify, request
from utils import Fingerprints

logger = logging.getLogger("legacy_qsar_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = Flask(__name__)


class LegacyQSAR:
    """
    This particular class uses the QSAR model from Lib-INVENT in a stand alone environment to avoid conflict dependencies.
    """

    def __init__(self, model_path: os.PathLike, fp="ECFP6", nBits=2048, **kwargs):
        """
        :param model_path: Path to pre-trained model (specifically clf.pkl)
        """
        self.model_path = model_path
        logger.debug(f"Loading model from {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.clf = pkl.load(f)
            logger.debug(f"Model loaded: {self.clf}")
        self.fp = fp
        self.nBits = nBits


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    logger.debug("POST request received")
    smiles = request.get_json().get("smiles", [])
    results = [{"smiles": smi, "pred_proba": 0.0} for smi in smiles]
    # Compute fingerprints
    fps = [
        Fingerprints.get(smi, name=model.fp, nBits=model.nBits, asarray=True)
        for smi in smiles
    ]
    # Track valid ones
    valid = []
    _ = [valid.append(i) for i, fp in enumerate(fps) if fp is not None]
    fps = [fp for fp in fps if fp is not None]
    # Compute predicted probability
    if len(fps):
        y_probs = model.clf.predict_proba(np.asarray(fps))[:, 1]
    else:
        y_probs = []
    for i, prob in zip(valid, y_probs):
        results[i].update({"pred_proba": float(prob)})
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument(
        "--model_path", type=str, help="Path to pre-trained model (e.g., clf.pkl)"
    )
    parser.add_argument("--fp", type=str, default="ECFP6", help="Fingerprint type")
    parser.add_argument(
        "--nBits", type=int, default=2048, help="Number of bits for fingerprint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = LegacyQSAR(**args.__dict__)
    app.run(port=args.port)
