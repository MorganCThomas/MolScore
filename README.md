# MolScore: A scoring, evaluation and benchmarking framework for de novo drug design
![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/GraphAbv2.png?raw=True)
## Overview

MolScore contains code to score and benchmark *de novo* compounds in the context of generative *de novo* design by generative models via the subpackage `molscore`, as well as, facilitate downstream evaluation via the subpackage `moleval`. An objective is defined via a JSON file which can be shared to propose new benchmark objectives, or to conduct multi-parameter objectives for drug design. Generative models with MolScore integrated can be found [here](https://github.com/MorganCThomas/MolScore_examples). 

Contributions and/or ideas for added functionality are welcomed!


## Installation
MolScore can be installed by cloning this repository and setting up an environment using your favourite environment manager (I recommend [mamba](https://github.com/conda-forge/miniforge#mambaforge)).

    git clone https://github.com/MorganCThomas/MolScore.git
    cd MolScore
    mamba env create -f environment.yml
    mamba activate molscore
    pip install ./

**Note:** You can use `pip install -e ./` to install in develop mode.

Alternatively, MolScore is available via the Python Package Index.

    pip install molscore --upgrade


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


## Usage
For further details, we refer you to the [tutorials](tutorials). Here is a snapshot of using MolScore with the GUIs available.

Here is a GIF demonstrating writing a config file with the help of the GUI, running MolScore in a mock example (scoring randomly sampled SMILES), and monitoring the output with another GUI.

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/molscore_demo.gif)

Once `molscore` has been implemented into a generative model, the objective needs to be defined! Writing a JSON file is a pain though so instead a streamlit app is provided do help. Simply call `molscore_config` from the command line (a simple wrapper to `streamlit run molscore/gui/config.py`)

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/config_v1_albuterol.png?raw=True)

Once the configuration file is saved, simply point to this file path and run *de novo* molecule optimization. If running with the monitor app you'll be able to investigate molecules as they're being generated. Simply call `molscore_monitor` from the command line (a wrapper to `streamlit run molscore/gui/monitor.py`).

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/monitor_v1_5HT2A_main.png?raw=True)

## Citation & Publications
If you use this software, please cite it as below.

    @article{thomas2023molscore,
    title={MolScore: A scoring and evaluation framework for de novo drug design},
    author={Thomas, Morgan and O'Boyle, Noel M and Bender, Andreas and de Graaf, Chris},
    journal={ChemRxiv},
    year={2023}
    }

This software was also utilised in the following publications:
1. **Thomas, M., Smith, R.T., O’Boyle, N.M. et al. Comparison of structure- and ligand-based scoring functions for deep generative models: a GPCR case study. J Cheminform 13, 39 (2021). https://doi.org/10.1186/s13321-021-00516-0**
2. **Thomas M, O'Boyle NM, Bender A, de Graaf C. Augmented Hill-Climb increases reinforcement learning efficiency for language-based de novo molecule generation. J Cheminform 14, 68 (2022).  https://doi.org/10.1186/s13321-022-00646-z**
3. **Handa K, Thomas M, Kageyama M, Iijima T, Bender A. On the difficulty of validating molecular generative models realistically: a case study on public and proprietary data. J Cheminform 15, 112 (2023). https://doi.org/10.1186/s13321-023-00781-1**
4. **Thomas M, Ahmad M, Tresadern G, de Fabritiis G. PromptSMILES: Prompting for scaffold decoration and fragment linking in chemical language models. ChemRxiv (2024). https://doi.org/10.26434/chemrxiv-2024-z5jnt/**
