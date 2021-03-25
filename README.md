# MolScore: An automated scoring function to facilitate and standardize evaluation of goal-directed generative models for de novo molecular design

WARNING: This codebase is still a work in progress and has not been robustly tested. (There's a long TODO list and writing tests is on it!)

## Overview

The aim of this codebase is to simply and flexibly 
automate the scoring of de novo compounds in generative models.
In doing so, also track and record de novo compounds for automated evaluation downstream.

- A central `molscore` class that is instantiated with a 
configuration file defining the scoring function parameters 
  (static configuration files enables reproducible tasks).
- When called, the instance 
accepts a list of SMILES and returns a list of scores. 
- The `molscore` class saves compounds into a dataframe for live monitoring with a dashboard
- A standardized dataframe allows automated downstream evaluation with `moleval/scripts/moses_n_statistics.py`

This requires only 3 lines of code to integrate into generative models.

The framework is also **designed to make it as simple as possible to integrate custom scoring functions**, for further information read the `./molscore/scoring_functions/README.md`.

Contributions and/or ideas for added functionality are welcomed! 

## Functionality

Some functionality has been adapted from other sources, so special mentions to:
* [GuacaMol](https://github.com/BenevolentAI/guacamol)
* [MOSES](https://github.com/molecularsets/moses)
* [REINVENT v2.0](https://github.com/MolecularAI/Reinvent)
* [reinvent-memory](https://github.com/tblaschke/reinvent-memory)

The current functionality included in this codebase includes:
* Scoring functions
  * Glide docking \[requires Schrodinger and licence]
  * Smina docking
  * ROCS shape overlay \[requires OpenEye and licence]
  * Glide docking from a ROCS overlay \[requires OpenEye and licence]
  * Openeye (FRED) docking (not tested) \[requires OpenEye and licence]
  * Scikit-learn QSAR models 
  * Substructure matches
  * Substructure filters
  * Tanimoto similarity
  * RDKit Descriptors
  
* Score modifiers
  * Linear transformation
  * Linear threshold transformation
  * Gaussian transformation
  * Step transformation
 
* Score methods
  * Arithmetic mean
  * Geometric mean
  * Weighted sum
  * Pareto pairs ([coming soon ...](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00517))
  
* [Diversity filters](https://github.com/tblaschke/reinvent-memory)
  * IdenticalTopologicalScaffold
  * CompoundSimilarity
  * IdenticalMurckoScaffold
  * ScaffoldSimilarity

## Installation

Conda is recommended to install this codebase, first clone this repo:

`git clone https://github.com/MorganCThomas/MolScore.git`

Move inside the directory:

`cd MolScore`

Now create a new conda environment using the .yml file:

`conda env create -f molscore_environment.yml`

Activate the new conda environment:

`conda activate molscore`

Next install the `molscore` package (I'd recommend `develop` instead of `install`):

`python setup_molscore.py develop`

## Configuration

An app can be used to help write the configuration file.

`streamlit run molscore/config_generator.py`

## Usage

For a toy demonstration, after installation of `molscore` and activation of the environment with `conda activate molscore`, open the python console (`python` or `ipython`). In the python console, run the following code:

```python
from molscore.test import MockGenerator
from molscore.manager import MolScore

# Here we setup a mock generator that simply samples molecules from a smiles file.
mg = MockGenerator()

# Now we setup MolScore passing in the configuration file
ms = MolScore(model_name='test',
              task_config='molscore/test/configs/test_qed.json')

# Now to simulate the scoring of a generative model, we'll pass in 100 molecules 10 times (e.g. batch size 100, iterations 10)
for i in range(10):
    ms(mg.sample(100))
    
# Finished mock generative model, wrap things up
ms.write_scores()
ms.kill_dash_monitor()
```

**Important** the MolScore class doesn't save the final dataframe until told to do so
with ms.write_scores(). This saves crucial time (which really does make a difference)
reading and writing from a .csv each iteration. During development, other formats were
explored such as an SQL database and parallelised dask dataframes, however, it was found
pandas was much quicker and parallelisation unnecessary, the dataframe shouldn't get so
large it's a problem for memory. If it does - the generative model should be more efficient!
Neither does the class close the dash monitor without calling ms.kill_dash_monitor()
(as it is run as a subprocess).
