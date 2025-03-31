# MolScore: A scoring, evaluation and benchmarking framework for de novo drug design
[![PyPI version](https://badge.fury.io/py/MolScore.svg)](https://badge.fury.io/py/MolScore)
[![PyPI Downloads](https://static.pepy.tech/badge/molscore)](https://pepy.tech/projects/molscore)
[![DOI](https://zenodo.org/badge/311350553.svg)](https://doi.org/10.5281/zenodo.14998608)

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/GraphAbv2.png?raw=True)
## Overview

[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w) | 
[Tutorials](tutorials) | 
[Examples](https://github.com/MorganCThomas/MolScore_examples) |
[Demo](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/molscore_demo.gif)

MolScore contains code to score *de novo* compounds in the context of generative *de novo* design by generative models via the subpackage `molscore`, as well as, facilitate downstream evaluation via the subpackage `moleval`. An objective is defined via a JSON file which can be shared to propose new multi-parameter objectives for drug design. MolScore can be used in several ways:
1. To implement a multi-parameter objective to for prospective drug design.
2. To benchmark objectives/generative models/optimization using benchmark mode (MolScoreBenchmark).
3. To implement a sequence of objectives using curriculum mode (MolScoreCurriculum).

Contributions and/or ideas for added functionality are welcomed!

## Installation
Install MolScore with PyPI (recommended):

    pip install molscore --upgrade

or directly from GitHub:

    git clone https://github.com/MorganCThomas/MolScore.git
    cd MolScore ; pip install -e .

> Note: I recommend mamba for environment handling

## Scoring

Simplest integration of MolScore requires a config file, for example:
```python
from molscore import MolScore
ms = MolScore(
    model_name='test',
    task_config='molscore/configs/QED.json',
    budget=10000
)
while not ms.finished:
    scores = ms.score(SMILES)
```

> Note: see [tutorial](tutorials/implementing_molscore.md#single-mode) for more detail

A GUI exists to help write the config file, which can be run with the following command.

    molscore_config

> Note: see [tutorial](tutorials/defining_an_objective.md) for more detail

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/config_v1_albuterol.png?raw=True)

<details>
  <summary><strong>Scoring functionality</strong></summary>
  &nbsp; <!-- This adds a non-breaking space for some spacing -->
  
  **Scoring functions**
  - **Descriptors**: RDKit, Maximum consecutive rotatable bonds, Penalized LogP, LinkerDescriptors (Fragment linking), 
    - [MolSkill](https://doi.org/10.1038/s41467-023-42242-1): Extracting medicinal chemistry intuition via preference machine learning as available on Nature Communications.
  - **Synthesizability**: [RAscore](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05401a), [AiZynthFinder](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1), SAscore, ReactionFilters (Scaffold decoration)
  - **2D Similarity**: Fingerprint similarity (any RDKit fingerprint and similarity measure), substructure match/filter, [Applicability domain](https://chemrxiv.org/engage/chemrxiv/article-details/625fc258bdc9c240d1dc12bb)
  - **3D Similarity**: ROCS, Open3DAlign
  - **QSAR**: Scikit-learn (classification/regression), [ChemProp](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
    - [PIDGINv5](https://zenodo.org/record/7547691#.ZCcLyo7MIhQ): Pre-trained RF classifiers for ~2,300 ChEMBL31 targets at different activity thresholds of 0.1 uM, 1 uM, 10 uM & 100 uM.
    - [ADMET-AI](https://www.biorxiv.org/content/10.1101/2023.12.28.573531v1): Pre-trained predictive models of various ADMET endpoints.
  - **Docking**: Glide<sup>a</sup>, Smina, OpenEye<sup>a</sup>, GOLD<sup>a</sup>, PLANTS, rDock, Vina, Gnina
    - **Ligand preparation**: RDKit->Epik, Moka->Corina, Ligprep, [Gypsum-DL](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0358-3)

 > <sup>a</sup> Requires a license

  **Transformation functions (transform values to [0-1])**
  - Linear
  - Linear threshold
  - Step
  - Step threshold
  - Gaussian

  **Aggregation functions (combine multiple scores into 1)**
  - Arithmetic mean
  - Geometric mean
  - Weighted sum
  - Weighted product
  - [Auto-weighted sum/product](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)
  - [Pareto front](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)

  **Filters (applied to final aggregated score)**
  - Any scoring function as a filter
  - Diversity filters
    - Unique
    - [Occurence](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z)
    - [Memory assisted](https://github.com/tblaschke/reinvent-memory)
      - [ScaffoldSimilarityECFP](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z)

</details>

## Benchmarking

Benchmarks are lists of objectives (configuration files) with metrics calculated upon exit. Re-implementations of existing benchmarks are available as presets.
```python
from molscore import MolScoreBenchmark

msb = MolScoreBenchmark(
    model_name='test',
    output_dir='./',
    benchmark='GuacaMol',
    budget=10000
)
for task in msb:
    while not task.finished:
        scores = task.score(SMILES)
```

Current benchmarks available include: [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839), [GuacaMol_Scaffold](https://arxiv.org/abs/2103.03864), [MolOpt](https://arxiv.org/pdf/2206.12411), [5HT2A_PhysChem, 5HT2A_Selectivity, '5HT2A_Docking'](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w), [LibINVENT Exp1&3](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00469), [MolExp(L)](https://arxiv.org/abs/2501.19153)

 > Note: inspect preset benchmarks with `MolScoreBenchmark.presets.keys()`

 > Note: see [tutorial](tutorials/implementing_molscore.md#benchmark-mode) for more detail

## Evaluation

The `moleval` subpackage can be used to calculate metrics for an arbitrary set of molecules.

```python
from moleval.metrics.metrics import GetMetrics

MetricEngine = GetMetrics(
    test=TEST_SMILES, # Model training data subset
    train=TRAIN_SMILES, # Model training data
    target=TARGET_SMILES, # Exemplary target data
)
metrics = MetricEngine.calculate(
    GEN_SMILES, # Generated data
)
```

 > Note: see [tutorial](tutorials/evaluating_molecules.md) for more detail

<details>
  <summary><strong>Metrics available</strong></summary>
  &nbsp; <!-- This adds a non-breaking space for some spacing -->

  **Intrinsice metrics (generated molecules only)**
  - Validity, Uniqueness, Scaffold uniqueness, Internal diversity (1 & 2), Scaffold diversity
  - [Sphere exclusion diversity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0): Measure of chemical space coverage at a specific Tanimoto similarity threshold. I.e., A score 0.5 indicates 50% of the sample size sufficiently describes the chemical space, therefore the higher the metric the more diverse the sample. Also see [here](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00519)
  - [Solow Polasky diversity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9) 
  - [Functional group diversity](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01328)
  - [Ring system diversity](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01328)
  - [Filters](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full): Passing of a set of drug-like filters (MolWt, Rotatable bonds, LogP etc.), Medicinal Chemistry substructures and PAINS substructures.
  - [Purchasability](https://github.com/whitead/molbloom): Molbloom prediction of presence in ZINC20

  **Extrinsic metrics (comparison to reference molecules)**
  - Novelty
  - [FCD](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234)
  - [Analogue similarity](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0): Proportion of generated molecules that are analogues to molecules in reference data.
  - [Analogue coverage](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0): Proportion of reference data that are analogues to generated data.
  - Functional group similarity
  - Ring system similarity
  - Single nearest neighbour similarity
  - Fragment similarity
  - Scaffold similarity
  - Outlier bits ([Silliness](https://github.com/PatWalters/silly_walks)): Average proportion of fingerprint bits (atomic environments) present in a generated molecule, not present anywhere in the reference data. The lower the silliness the better.
  - Wasserstein distance (LogP, SA Score, NP score, QED, Weight)

</details>

## Additional functionality

- Curriculum learning (see [tutorial](tutorials/implementing_molscore.md#curriculum-mode))
- Experience replay buffers (see [tutorial](tutorials/implementing_molscore.md#using-a-replay-buffer))
- Parallelisation (see [tutorial](tutorials/parallelisation.md))
- A GUI for monitoring generated molecules (see below)

    ```molscore_monitor```

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/monitor_v1_5HT2A_main.png?raw=True)

## Citation & Publications
If you use this software, please cite it as below.

    @article{thomas2024molscore,
    title={MolScore: a scoring, evaluation and benchmarking framework for generative models in de novo drug design},
    author={Thomas, Morgan and O’Boyle, Noel M and Bender, Andreas and De Graaf, Chris},
    journal={Journal of Cheminformatics},
    volume={16},
    year={2024},
    publisher={BMC}
    }

This software was also utilised in the following publications:
1. **Thomas, M., Smith, R.T., O’Boyle, N.M. et al. Comparison of structure- and ligand-based scoring functions for deep generative models: a GPCR case study. J Cheminform 13, 39 (2021). https://doi.org/10.1186/s13321-021-00516-0**
2. **Thomas M, O'Boyle NM, Bender A, de Graaf C. Augmented Hill-Climb increases reinforcement learning efficiency for language-based de novo molecule generation. J Cheminform 14, 68 (2022).  https://doi.org/10.1186/s13321-022-00646-z**
3. **Handa K, Thomas M, Kageyama M, Iijima T, Bender A. On the difficulty of validating molecular generative models realistically: a case study on public and proprietary data. J Cheminform 15, 112 (2023). https://doi.org/10.1186/s13321-023-00781-1**
4. **Thomas M, Ahmad M, Tresadern G, de Fabritiis G. PromptSMILES: Prompting for scaffold decoration and fragment linking in chemical language models. J Cheminform 16, 77 (2024). https://doi.org/10.1186/s13321-024-00866-5**
5. **Bou A, Thomas M, Dittert S, Ramírez CN, Majewski M, Wang Y, Patel S, Tresadern G, Ahmad M, Moens V, Sherman W. ACEGEN: Reinforcement learning of generative chemical agents for drug discovery. J Chem Inf Model 64, 15 (2024). https://doi.org/10.1021/acs.jcim.4c00895**
6. **Thomas M, Matricon PG, Gillespie RJ, Napiórkowska M, Neale H, Mason JS, Brown J, Fieldhouse C, Swain NA, Geng T, O'Boyle NM. Modern hit-finding with structure-guided de novo design: identification of novel nanomolar adenosine A2A receptor ligands using reinforcement learning. ChemRxiv (2024) https://doi.org/10.26434/chemrxiv-2024-wh7zw-v2**
