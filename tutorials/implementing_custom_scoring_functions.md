# Implementing custom scoring functions

It is very likely that you have your own idea or code for a scoring function, and you want to make use of everything else MolScore has. This can be done by writing your own scoring function class, and adding it to an be available for MolScore.

## Writing your own scoring function
Firstly, a scoring function in MolScore must be a class with a \_\_call\_\_ method that takes a list of SMILES, and returns a list of dictionaries in the same order, containing the results. 

Secondly, if the scoring function is properly annotated, it can be automatically parsed and used in the molscore GUI when defining configs. 

The base format can be found at `molscore/scoring_functions/base.py`. Here is an example to calculate the QED of a molecule.

```python
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import QED

class QED:
    """Calculate QED"""  # Description used in GUI
    return_metrics = ["QED"]  # Endpoints that can be selected in GUI
    
    def __init__(self, prefix: str, **kwargs):
        """
        :param prefix: Description
        """
        # Full typing should be used (to define GUI widget type), PyCharm style docstring for parameters only (to add GUI description for parameters), and choices can be specified in square brackets, for example, [Choice 1, Choice 2, Choice 3] resulting in a dropdown list in the GUI. Hence, avoid the use of square brackets otherwise.
        self.prefix = prefix.strip().replace(' ', '_')

    @staticmethod
    def calculate_QED(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return QED(mol)
        else:
            return 0.0
    
    def __call__(self, smiles: list, **kwargs) -> List[Dict]:
        results = []
        for smi in smiles:
            results.append({'smiles': smi, f'{self.prefix}_QED': self.calculate_QED(smi)})
        # Here results should contain every smiles and parameter specified in return_metrics with the specified prefix.
        return results
```
The class should be able to handle invalid smiles/molecules and/or expected errors. In this case, the corresponding dictionary should set all values to 0.0 (except for 'smiles').

## Adding the scoring function to MolScore
To integrate the new scoring function, it needs to be imported in `molscore/scoring_functions/__init__.py` file and added to the `all_scoring_functions` list. I recommend adding it as such so that if any errors occur, you can still use the rest of MolScore and an message will be printed.

```python
try:
    from molscore.scoring_functions.qed import QED
    all_scoring_functions.append(QED)
except Exception:
    logger.warn(f"QED: currently unavailable due to the following: {e}")
```

## Advanced: Writing a server scoring function to be run from a dedicated python environment
Now, for many reasons, scoring function code may need to be run from a different, dedicated, fixed python environment. One option to integrate this is to write a server that runs in that environment that interacts with MolScore.

### Server side
Here we write a scoring function server to be run from it's own python environment using flask (hence, flask must be installed in the environment), an example format can be found at `molscore/scoring_functions/servers/base.py` mostly related to loading a QSAR model and running it. For easiest integration, the module should be added to `molscore/scoring_functions/servers` directory. Below is an example for running QED (again).

```python
import argparse

from rdkit.Chem import QED
from rdkit.Chem import AllChem as Chem
from flask import Flask, request, jsonify

app = Flask(__name__)

def calculate_QED(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return QED(mol)
    else:
        return 0.0

@app.route('/', methods=['POST'])
def compute():
    smiles = request.get_json().get('smiles', [])

    results = []
    for smi in smiles: 
        results.append({'smiles': smi, 'QED': calculate_QED(smi)})
    return jsonify(results)

def get_args():
    parser = argparse.ArgumentParser(description='Run a scoring function server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    app.run(port=args.port)
```

### Client side (i.e., MolScore scoring function)
Now, we need to write a client side scoring function to send SMILES to our server and receive the results. An example `BaseServerSF` can be found at `molscore/scoring_functions/base.py` and can be inherited to deal with most of the automation. First, `BaseServerSF` will check if a named environment exists, if not, it will try to install it if provided a `environment.yaml`. Then, it will launch the server via the python environment. Then it will send SMILES to the server, receive the results, and add the prefix. 

Continueing on from our QED example...

```python
from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

class QED(BaseServerSF):

    return_metrics = ["QED"]

    def __init__(
        self,
        prefix: str = "example",
        env_engine: str = "mamba",
        **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., test)
        :param env_envine: Environment engine [conda, mamba]
        """
        super().__init__(
            prefix=prefix, # This will be automatically added to endpoints received
            env_engine=env_engine,  # Which to use, I recommend mamba
            env_name="qed",  # The name of the python environment
            env_path="qed.yaml",  # A yaml file defining the environment
            server_path=resources.files("molscore.scoring_functions.servers").joinpath("name_of_server.py"), # Or absolute path
            server_grace=30,  # How long to leave the server to setup
            server_kwargs={},  # Keyword arguments passed to the server via the command line
            )
```

This can then be added to `molscore/scoring_functions/__init__.py` as described [before](#adding-the-scoring-function-to-molscore).