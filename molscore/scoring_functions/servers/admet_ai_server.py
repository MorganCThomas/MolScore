import argparse
import logging
from flask import Flask, request, jsonify

from admet_ai import ADMETModel

logger = logging.getLogger("admet_ai_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = Flask(__name__)


class AdmetAI:
    """
    This particular class uses the QSAR model from Lib-INVENT in a stand alone environment to avoid conflict dependencies.
    """

    def __init__(self, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        """
        self.ADMETModel = ADMETModel(include_physchem=False, cache_molecules=False)


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    logger.debug("POST request received")
    smiles = request.get_json().get("smiles", [])
    logger.debug(f"Read SMILES:\n\t{smiles}")

    # Make predictions
    logger.debug("Making predictions")
    preds = model.ADMETModel.predict(smiles=smiles)
    preds = preds.to_dict("records")
    logger.debug(f"Predictions received:\n\t{preds}")

    # Admet_AI removes invalids, so we need to check they're all returned
    assert len(smiles) == len(preds)

    # Update dictionary
    results = []
    for smiles, pred in zip(smiles, preds):
        r = {"smiles": smiles}
        r.update(pred)
        results.append(r)

    # Return results
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description="Run a scoring function server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = AdmetAI(**args.__dict__)
    app.run(port=args.port)
