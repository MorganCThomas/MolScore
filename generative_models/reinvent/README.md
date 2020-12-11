
# REINVENT
## This is a fork of the original code for use with the MolScore framework
See "Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning" and [Release v1.0.1](https://github.com/MarcusOlivecrona/REINVENT/releases/tag/v1.0.1) for reference.

## Installation

After installing `molscore`, extra requirements can be satisfied by running:

`sh requirements.sh`

Unfortunately this is a rather hacky way to satisfy old dependencies including a hot fix for scikit-learn (version required to run the SVM model found in `data`) and downloading of previous pytorch versions (relies on `wget`).

All other requirements should be satisfied by the molscore environment setup previously.

## Modifications

The only modifications to the original code is adding argparse options for more user friendly runnning from the command line for the `train_agent.py`, `train_prior.py` and `data_structs.py` files, and addition of `sample.py` for sampling from a trained model. `train_agent.py` has been modified to accept a molscore config file to control the scoring function. 

The minimum requirements to integrate the `molscore` scoring function framework with this generative model was 3 lines of code:

* `from molscore.manager import MolScore`
* `scoring_function = MolScore(config=<path to config>)`
* Remove previous definition of scoring function.

## Usage

To train a Prior starting with a SMILES file (text file of smiles seperated by new lines \[no headers, or other fields\]):

* First filter the SMILES and construct a vocabulary from the remaining sequences (run `data_structs.py --help`).

* Then use `train_prior.py` to train the Prior on the filtered smiles and Vocabulary returned by `data_structs.py`.

To train an Agent using our Prior, don't use the `main.py` script! Instead, use the `train_agent.py` script which accepts a molscore config files as input.

Training can be visualized via molscore with the dash monitor, that should automatically provide a link (if set to true in the `molscore` config).


