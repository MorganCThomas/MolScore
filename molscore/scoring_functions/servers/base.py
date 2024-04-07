import argparse
from flask import Flask, request, jsonify


app = Flask(__name__)


class Model:
    """
    Create a Model object that loads models once, for example.
    """

    def __init__(self, **kwargs):
        pass
        # TODO Load a model and anything necessary attributes here


@app.route("/", methods=["POST"])
def compute():
    # Get smiles from request
    smiles = request.get_json().get("smiles", [])

    # TODO Make predictions
    preds = model.predict(smiles=smiles)

    # TODO Update dictionary
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
    # TODO Add more arguments here
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port)
