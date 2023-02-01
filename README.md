# MolScore: An automated scoring function to facilitate and standardize evaluation of goal-directed generative models for de novo molecular design

## Overview

The aim of this codebase is to simply and flexibly 
automate the scoring of *de novo* compounds in generative models (`molscore`).
In doing so, also track and record de novo compounds for automated evaluation downstream (`moleval`).

1. Install both python packages
2. Implement molscore in a generative model script (substituted for a scoring function)
3. Describe the desired objective with a configuration file that is passed to `molscore`
4. Run generative model optimization
5. Evaluate *de novo* molecules with `moleval`

The objective can be reproduced and shared via the configuration file to create new benchmarks, or to conduct multi-parameter objectives for AI discovery.

It is also easy to add new custom scoring function - see [here](https://github.com/MorganCThomas/MolScore/blob/main/molscore/scoring_functions/README.MD)

In total only **3 lines of code are needed** to implement into generative models (...althought 5 is preferable, sorry).

Contributions and/or ideas for added functionality are welcomed!

This code here was in the following publications:
1. **Thomas, M., Smith, R.T., O’Boyle, N.M. et al. Comparison of structure- and ligand-based scoring functions for deep generative models: a GPCR case study. J Cheminform 13, 39 (2021). https://doi.org/10.1186/s13321-021-00516-0**
2. **Thomas M, O'Boyle NM, Bender A, de Graaf C. Augmented Hill-Climb increases reinforcement learning efficiency for language-based de novo molecule generation. ChemRxiv.  https://chemrxiv.org/engage/chemrxiv/article-details/6253f5f66c989c04f6b40986**

## Functionality
* Scoring functions
  * Docking (parallelized via [Dask](https://docs.dask.org/en/latest/deploying-cli.html))
    * Glide docking \[licence required]
    * Glide docking from a ROCS overlay \[licence required]
    * Smina docking \[open-source]
    * Openeye-FRED docking \[licence required]
    * GOLD \[license required]
    * PLANTS \[license required]
  * Shape
    * ROCS shape overlay \[licence required]
    * Open 3D align \[open-source]
  * Ligand preparation
    * RDKit stereoenumeration and Epik \[licence required]
    * Moka and Corina \[licence required]
    * Ligprep \[licence required]
    * Gypsum-DL \[open-source]
  * Scikit-learn classification models (including pre-trained RF classifiers for ~2,300 ChEMBL31 targets PIDGINv5) 
  * Substructure matches
  * Substructure filters
  * Fingerprint similarity
  * RDKit Descriptors
  * [Retrosynthetic accessability score](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05401a)
  * [Applicability domain](https://chemrxiv.org/engage/chemrxiv/article-details/625fc258bdc9c240d1dc12bb)
  
* Score modifiers
  * Linear transformation
  * Linear threshold transformation
  * Gaussian transformation
  * Step transformation
 
* Score methods
  * Arithmetic mean
  * Geometric mean
  * Weighted sum
  * [Dynamic weighted sum](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)
  * [Pareto Front](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)
  
* Filters
  * Unique
  * Occurences
  * [Diversity filters](https://github.com/tblaschke/reinvent-memory)
  * ScaffoldSimilarity (with ECFP)
  * [Applicability domain](https://chemrxiv.org/engage/chemrxiv/article-details/625fc258bdc9c240d1dc12bb) (coming soon ...)
  
* Performance metrics
  * [GuacaMol metrics](https://github.com/BenevolentAI/guacamol)
  * [MOSES metrics](https://github.com/molecularsets/moses)
  * [Sphere exclusion diversity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0)
  * [Functional groups and ring systems](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01328)
  * [Silly molecules](https://github.com/PatWalters/silly_walks)
  * [Analogue similarity](https://github.com/tblaschke/reinvent-memory)
  * [Analogue coverage](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0)

## Installation

To install molscore:

```
git clone https://github.com/MorganCThomas/MolScore.git
cd MolScore
conda env create -f molscore_environment.yml
conda activate molscore
python setup_molscore.py develop
```

To install moleval:
```
conda env create -f moleval_environment.yml
conda activate moleval
python setup_moleval.py develop
```

## Implementation

Implementing `molscore` is as simple as importing it, instantiating it (pointing to the configuration file) and then scoring molecules. This should easily fit into most generative model pipelines.

```python
from molscore.manager import MolScore

# Instantiate MolScore and point to configuration file
ms = MolScore(model_name='test',
              task_config='molscore/configs/QED.json')
              
# Calling it simply scores a list of smiles (SMILES)
scores = ms(SMILES)
    
# When finished, all recorded smiles need to be saved and the live monitor killed
ms.write_scores()
ms.kill_monitor()
```

## Usage

First we need to decide the objective by writing a configuration json file. What a pain!

Instead let's use an app to write it for us! This app contains parameters, their descriptions and defaults.

```
streamlit run molscore/config_generator.py
```

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/config_generator_1.png?raw=True)

Then simply point use this configuration file and run *de novo* molecule optimization. If running with the live monitor you'll be able to investigate molecules as they're being generated.

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/st_monitor_2.png?raw=True)



