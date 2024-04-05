# MolScore: A scoring and evaluation framework for de novo drug design
![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/GraphAbv2.png?raw=True)
## Overview

The aim of this codebase is to simply and flexibly 
automate the scoring of *de novo* compounds in generative models via the subpackage `molscore`. As well as, facilitate evaluation downstream via the subpackage `moleval`. An objective is designed via a JSON file which can be shared to propose new benchmark objectives, or to conduct multi-parameter objectives for drug design.

Custom scoring functions can be implemented following the guidelines [here](https://github.com/MorganCThomas/MolScore/blob/main/molscore/scoring_functions/README.MD)

Contributions and/or ideas for added functionality are welcomed!

A description of this software:

**Thomas, M., O'Boyle, N.M., Bender, A., de Graaf, C. MolScore: A scoring and evaluation framework for de novo drug design. chemRxiv (2023). https://doi.org/10.26434/chemrxiv-2023-c4867**

This code here was used in the following publications:
1. **Thomas, M., Smith, R.T., O’Boyle, N.M. et al. Comparison of structure- and ligand-based scoring functions for deep generative models: a GPCR case study. J Cheminform 13, 39 (2021). https://doi.org/10.1186/s13321-021-00516-0**
2. **Thomas M, O'Boyle NM, Bender A, de Graaf C. Augmented Hill-Climb increases reinforcement learning efficiency for language-based de novo molecule generation. J Cheminform 14, 68 (2022).  https://doi.org/10.1186/s13321-022-00646-z**

## Installation
Mamba should be used to install the molscore environment as it is considerably better than conda. If you do not have mamba first install this package manager following the instructions [here](https://github.com/conda-forge/miniforge#mambaforge).
```
git clone https://github.com/MorganCThomas/MolScore.git
cd MolScore
mamba env create -f environment.yml
mamba activate molscore
python setup.py develop
```
**Note:** Depending if you have conda already installed, you may have to use `conda activate` instead and point to the env path directly for example, `conda activate ~/mambaforge/envs/molscore`

## Implementation into a generative model

Implementing `molscore` is as simple as importing it, instantiating it (pointing to the configuration file) and then scoring molecules. This should easily fit into most generative model pipelines.

```python
from molscore import MolScore

# Instantiate MolScore, assign the model name and point to configuration file describing the objective
ms = MolScore(model_name='test', task_config='molscore/configs/QED.json')
              
# Calling it simply scores a list of smiles (SMILES) - to be integrated into a for loop during model optimization
scores = ms.score(SMILES)
    
# When the program exits, all recorded smiles will be saved and the monitor app (if selected) will be closed
```
**Note**: Other MolScore parameters include `output_dir` to override any specified in the `task_config`.

Alternatively, a can be set `budget` to specify the maximum number of molecules to score, after the budget is reached `ms.finished` will be set to `True` which can be evaluated to decide when to exit an optimization loop. For example,

```python
from molscore import MolScore
ms = MolScore(model_name='test', task_config='molscore/configs/QED.json', budget=10000)
while not ms.finished:
    scores = ms.score(SMILES)
```

A benchmark mode is also available that can be used to iterate over a selection of tasks defined in config files, or a set of pre-defined benchmarks that come packaged with MolScore including [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839), [GuacaMol_Scaffold](https://arxiv.org/pdf/2103.03864.pdf), [MolOpt](https://arxiv.org/abs/2206.12411), [5HT2A_PhysChem](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda), [5HT2A_Selectivity](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda), [5HT2A_Docking](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda), [LibINVENT_Exp1](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00469), [LinkINVENT_Exp3](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00115b).

```python
from molscore import MolScoreBenchmark

# As an example, configs re-implementing GuacaMol are available as a preset benchmark, or custom tasks can be provided 
msb = MolScoreBenchmark(model_name='test', benchmark='GuacaMol', budget=10000)
for task in msb:
    # < Initialize generative model >
    while not task.finished:
        # < Sample smiles from generative model >
        scores = task.score(smiles)
        # < Update generative model >
# When the program exits, a summary of performance will be saved
```

**Note: A generative language model with MolScore already implemented can be found [here](https://github.com/MorganCThomas/SMILES-RNN).**

## Usage
Here is a GIF demonstrating writing a config file with the help of the GUI, running MolScore in a mock example (scoring randomly sampled SMILES), and monitoring the output with another GUI.

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/molscore_demo.gif)

Once `molscore` has been implemented into a generative model, the objective needs to be defined! Writing a JSON file is a pain though so instead a streamlit app is provided do help. Simply call `molscore_config` from the command line (a simple wrapper to `streamlit run molscore/gui/config.py`)

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/config_v1_albuterol.png?raw=True)

Once the configuration file is saved, simply point to this file path and run *de novo* molecule optimization. If running with the monitor app you'll be able to investigate molecules as they're being generated. Simply call `molscore_monitor` from the command line (a wrapper to `streamlit run molscore/gui/monitor.py`).

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/monitor_v1_5HT2A_main.png?raw=True)

## Functionality
Scoring functionality present in **molscore**, some scoring functions require external softwares and necessary licenses.  

|Type|Method|
|---|---|
|Docking|Glide, Smina, OpenEye, GOLD, PLANTS, rDock, Vina, Gnina|
|Ligand preparation|RDKit->Epik, Moka->Corina, Ligprep, [Gypsum-DL](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0358-3)|
|3D Similarity|ROCS, Open3DAlign|
|2D Similarity|Fingerprint similarity (any RDKit fingerprint and similarity measure), substructure match/filter, [Applicability domain](https://chemrxiv.org/engage/chemrxiv/article-details/625fc258bdc9c240d1dc12bb)|
|Predictive models|Scikit-learn (classification/regression), [PIDGINv5](https://zenodo.org/record/7547691#.ZCcLyo7MIhQ)<sup>a</sup>, [ChemProp](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237), [ADMET-AI](https://www.biorxiv.org/content/10.1101/2023.12.28.573531v1)|
|Synthesizability|[RAscore](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05401a), [AiZynthFinder](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1), SAscore, ReactionFilters (Scaffold decoration)|
|Descriptors|RDKit, Maximum consecutive rotatable bonds, Penalized LogP, LinkerDescriptors (Fragment linking) etc.|
|Transformation methods|Linear, linear threshold, step threshold, Gaussian|
|Aggregation methods|Arithmetic mean, geometric mean, weighted sum, product, weighted product, [auto-weighted sum/product, pareto front](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)|
|Diversity filters|Unique, Occurence, [memory assisted](https://github.com/tblaschke/reinvent-memory) + ScaffoldSimilarityECFP|

<sup>a</sup> PIDGINv5 is a suite of pre-trained RF classifiers on ~2,300 ChEMBL31 targets
  
Performance metrics present in **moleval**, many of which are from [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) or [MOSES](https://www.frontiersin.org/articles/10.3389/fphar.2020.565644/full). 
|Type|metric|
|---|---|
|Intrinsic property|Validity, Uniqueness, Scaffold uniqueness, Internal diversity (1 & 2), Sphere exclusion diversity<sup>b</sup>, Solow Polasky diversity<sup>g</sup>, Scaffold diversity, Functional group diversity<sup>c</sup>, Ring system diversity<sup>c</sup>, Filters (MCF & PAINS), Purchasability<sup>d</sup>|
|Extrinsic property<sup>a</sup>|Novelty, FCD, Analogue similarity<sup>e</sup>, Analogue coverage<sup>b</sup>, Functional group similarity, Ring system similarity, Single nearest neighbour similarity, Fragment similarity, Scaffold similarity, Outlier bits (Silliness)<sup>f</sup>, Wasserstein distance (LogP, SA Score, NP score, QED, Weight)|

<sup>a</sup> In reference to a specified external dataset  
<sup>b</sup> As in our previous work [here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0)  
<sup>c</sup> Adaption based on [Zhang et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01328)  
<sup>d</sup> Using [molbloom](https://github.com/whitead/molbloom)  
<sup>e</sup> Similar to [Blaschke et al.](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0)  
<sup>f</sup> Based on [SillyWalks](https://github.com/PatWalters/silly_walks) by Pat Walters  
<sup>g</sup> Based on [Liu et al.](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)

## Parallelisation
Most scoring functions implemented can be parallelised over multiple CPUs simply using pythons multiprocessing by specifying the `n_jobs` parameter. Some more computationally expensive scoring functions such as molecular docking are parallelised using a [Dask](https://www.dask.org/) to allow distributed parallelisation accross compute nodes (`cluster` parameter). Either supply the number of CPUs to utilize on a single compute node to the scheduler address setup via the [Dask CLI](https://docs.dask.org/en/latest/deploying-cli.html). 

To setup a dask cluster first start a scheduler by running (the scheduler address will be printed to the terminal)
```
mamba activate <env>
dask scheduler
```
Now to start workers accross multiple nodes, simply SSH to a connected node and run
```
mamba activate <env>
dask worker <scheduler_address> --nworkers <n_jobs> --nthreads 1
```
Repeat this for each node you wish to add to the cluster (ensure the conda environment and any other dependencies are loaded as you would normally). Then supply modify the config so that `cluster: <scheduler_address>`.

**Optional**: Sometimes you may not want to keep editing this parameter in the config file and so environment variables can be set which will override anything provided in the config. To do this, before running MolScore export either of the following variables respectively.
```
export MOLSCORE_NJOBS=<n_jobs>
export MOLSCORE_CLUSTER=<scheduler_address>
```
**Note**: It is recommended to not use more than the number of logical cores available on the a particular machine, for example, on a 12-core machine (6 logical cores hyperthreaded) I would not recommend more than 6 workers as it may overload CPU. 

## Tests
Some unittests are available.
```
cd molscore/tests
python -m unittest
```

Or any individual test, for example
```
python test_docking.py
```

Or, you can test a configuration file, for example
```
python test_configs.py <path to config1> <path to config2> <path to dir of configs>
```

