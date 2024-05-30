import argparse
import logging

from flask import Flask, jsonify, request
from molskill.scorer import MolSkillScorer
from utils import get_mol

logger = logging.getLogger("molskill_server")
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

    def __init__(self, **kwargs):
        # NOTE: num_workers > 1 leads to Boost picking error, multiprocessing doesn't seem to work
        self.scorer = MolSkillScorer(num_workers=0)


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    smiles = request.get_json().get("smiles", [])

    # Handle invalid as this is not handled by MolSkill
    valids = []
    for i, smi in enumerate(smiles):
        if get_mol(smi):
            valids.append(i)

    # Make predictions
    scores = model.scorer.score([smiles[i] for i in valids])

    # Update results
    results = [{"smiles": smi, "score": 0.0} for smi in smiles]
    for i, score in zip(valids, scores):
        results[i].update({"score": float(score)})

    # Return results
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    # TODO Add more arguments here
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port)
