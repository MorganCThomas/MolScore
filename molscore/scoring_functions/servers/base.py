import argparse
import logging
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)


class Model:
    """
    This particular class uses the QSAR model from Lib-INVENT in a stand alone environment to avoid conflict dependencies.
    """
    def __init__(self, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        """
        # TODO Load the model and anything necessary attributes here


@app.route('/', methods=['POST'])
def compute():
    # Get smiles from request
    logger.debug('POST request received')
    smiles = request.get_json().get('smiles', [])
    logger.debug(f'Read SMILES:\n\t{smiles}')

    # TODO Make predictions
    logger.debug('Making predictions')
    preds = model.predict(smiles=smiles)
    logger.debug(f'Predictions received:\n\t{preds}')
    
    # Admet_AI removes invalids, so we need to check they're all returned
    assert len(smiles) == len(preds)

    # TODO Update dictionary
    results = []
    for smiles, pred in zip(smiles, preds):
        r = {'smiles': smiles}
        r.update(pred)
        results.append(r)

    # Return results
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(description='Run a scoring function server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    # TODO Add more arguments here
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port)