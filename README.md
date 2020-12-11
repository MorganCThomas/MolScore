# MolScore: An automated scoring function to facilitate and standardize evaluation of goal-directed generative models for de novo molecular design

WARNING: This codebase is still a work in progress and has not been robustly tested. (There's a long TODO list and writing tests is on it!)

## Aim

This codebase is born out of frustration with the rapid pace, yet often brief evaluation (w.r.t chemistry) of generative models for de novo molecule design. It is designed to aid the comparison of generative models w.r.t scoring functions, model hyperparameters or different goal-directed generative model algorithms.

The aim of this codebase is to be integrable into the majority of goal-directed generative models that require an external scoring function. Thus the idea is conceptually simple, a central `molscore` class that when called, accepts a list of SMILES and returns a list of scores. In doing so, the `molscore` class also acts as a data manager, by storing all molecules sampled and scores generated. This allows for a number of benefits including:

* Data lookup so that duplicated SMILES don't require recalculation by a scoring function (assuming the score is static), saving time for expensive scoring functions.
* Live monitoring of intermediate molecules sampled during training to track the progress with respect to chemistry and score. 
* Standardized output format, facilitating automated evaluation of results with `moleval`.

The central `molscore` class only requires one parameter for setup and iniatilisation in the form of a config (.json) file. This allows for **sharing of configuratins for reproducible scoring functions** with respect to the functions themselves, score modifiers and aggregation methods for multi-parameter optimization. The framework is also **designed to make it as simple as possible to integrate custom scoring functions**, for further information read the `./molscore/scoring_functions/README.md`.

Contributions and/or ideas for added functionality is welcomed! In an ideal world, authors of new publications could integrate `molscore` to save time in scoring function setup and aid in their evaluation, but could likewise contribute new scoring functions and share the config files as benchmark/reproducible tasks. With the aim to build more complex benchmarking tasks for goal directed generative models. 

## MolScore functionality

Some functionality has been adapted from other sources, so special mentions to:
* [GaucaMol](https://github.com/BenevolentAI/guacamol)
* [MOSES](https://github.com/molecularsets/moses)
* [REINVENT v2.0](https://github.com/MolecularAI/Reinvent)
* [reinvent-memory](https://github.com/tblaschke/reinvent-memory)
* [smina-docking-benchmark](https://github.com/cieplinski-tobiasz/smina-docking-benchmark) (coming soon ...)

The current functionality included in this codebase includes:
* Scoring functions
  * Glide docking
  * ROCS shape overlay
  * Glide from ROCS best conformer overlay
  * Openeye (FRED) docking (not tested)
  * Smina docking (coming soon ...)
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
  
* Diversity filters
  * IdenticalTopologicalScaffold
  * CompoundSimilarity
  * IdenticalMurckoScaffold
  * ScaffoldSimilarity

## Installation

Conda is recommended to install this codebase, first clone this repo:

`git clone https://github.com/MorganCThomas/MolScore.git`

Now create a new conda environment using the .yml file:

`conda env create -f molscore_environment.yml`

Activate the new conda environment:

`conda activate molscore`

Next install the `molscore` package (if you plan on modification, I'd recommend `develop` instead of `install`):

`python setup_molscore.py install`

## Configuration

A template configuration file can be found `./molscore/configs/template.json`, which displays the majority of options. (A script to check or help write configuration files is also on the TODO list).

Briefly, the configuration file is structured as follows:

* logging (dict)
  * model
    * name: str # This is saved in the output .csv under column name 'model'
    * comments: # This is purely for self-reference, the config file gets saved with every use for logging purposes. E.g. {"batch_size": 64, "n_epoch": 12}
  * task
    * name: str # This is saved in the output .csv under column name 'task'
    * comments: # This is purely for self-reference e.g. "Optimization of DRD2 docking, using this pdb file... etc."
* output_dir: str # Path to output directory for saving files
* load_from_previous: bool # Wether to continue a previous run
* previous_dir: str # Only relevent if load 
* dash_monitor (dict)
  * run: bool (true/false)
  * pdb_path: \[null, str] # This is only for use with dash_monitor_3D, depends on dash_bio and scikit-learn which is not install by default*
* diversity_filter (dict)
  * run: bool
  * name: str # Must match class name of desired filter, list found in `./molscore/scaffold_memory/__init__.py`
  * parameters: dict # Set of parameters to be passed to diversity filter
* scoring_functions (list)
  * name: str # Must match class name of scoring function, list found in `./molscore/scoring_functions/__init__.py`
  * run: bool # Whether to run the scoring function or not can equally omit from list
  * parameters: dict # Parameters passed to initialize scoring function
    * prefix: str # This parameter should be required by all scoring functions classes to label metrics, enabling distinguishment of multiple scoring functions of the same type.
* scoring (dict)
  * method: str # Must match function name (e.g. wsum) found in `./molscore/utils/score_methods.py`
  * metrics (list)
    * name: str # Must match <{prefix}_{metric}> where prefix was used in the scoring function parameter and metric must be a known metric returned by the scoring function before hand
    * weight: float # A number between 0 and 1, only relevent when using wsum scoring method.
    * modifier: str # Must match name of modifier functions found in `./molscore/utils/score_modifiers.py`
    * parameters (dict)
      * objective: str # One of \[maximize, minimize, range]
      * other: any specific parameters required from score modifiers e.g. sigma, mu etc.

