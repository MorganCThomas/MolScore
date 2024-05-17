# Using moleval to evaluate de novo chemistry generated

There is currently two approaches to calculating metrics and evaluating the results of de novo molecule optimisation.

## Calculating performance

The first is to compute 'performance' (mostly with respect to sample efficiency) when a run has finished. 

This can be done by calling the `compute_metrics` method.

```python
from molscore import MolScore, MockGenerator

mg = MockGenerator()
SMILES = mg.sample(50)

ms = MolScore(model_name='test', task_config='molscore/configs/QED.json')
scores = ms.score(SMILES)

# Once finished
metrics = ms.compute_metrics(
    endpoints=None, # Optional list: by default will use the running final score/reward value
    thresholds=None,  # Optional list: if specified will calculate the yield of molecules above that threshold 
    chemistry_filters_basic=True,  # Optional, bool: Additionally re-calculate metrics after filtering out unreasonable chemistry
    budget=10000,  # Optional, int: Calculate metrics only with molecules within this budget
    n_jobs=1,  # Optional, int: Multiprocessing
    benchmark=None,  # Optional, str: Name of benchmark, this may specify additional metrics to compute
)
```
The output of this is a dictionary containing metrics. Inlcuding after application of Basic Chemistry Filters (B-CF).

    {'Top-1 Avg Score': 0.9211228031521655,
    'Top-10 Avg Score': 0.9057869800137961,
    'Top-100 Avg Score': 0.7462685674245637,
    'Top-1 AUC Score': 0.4605614015760828,
    'Top-10 AUC Score': 0.4605614015760828,
    'Top-100 AUC Score': 0.4605614015760828,
    'Yield Score': 1.0,
    'Yield AUC Score': 25.0,
    'Yield Scaffold Score': 0.94,
    'Yield AUC Scaffold Score': 23.5,
    'B-CF Top-1 Avg Score': 0.9211228031521655,
    'B-CF Top-10 Avg Score': 0.9057869800137961,
    'B-CF Top-100 Avg Score': 0.7462685674245637,
    'B-CF Top-1 AUC Score': 0.4605614015760828,
    'B-CF Top-10 AUC Score': 0.4605614015760828,
    'B-CF Top-100 AUC Score': 0.4605614015760828,
    'B-CF Yield Score': 1.0,
    'B-CF Yield AUC Score': 25.0,
    'B-CF Yield Scaffold Score': 0.94,
    'B-CF Yield AUC Scaffold Score': 23.5,
    'B-CF': 1.0}

**Note**: This is automatically run and saved for each task when using `MolScoreBenchmark` mode. 

## Calculating metrics

Second is another suite of chemistry related metrics more aimed at chemistry and comparison to training, validation, and target chemistry.

```python
from molscore import MockGenerator
from moleval.metrics.metrics import GetMetrics

mg = MockGenerator()
GEN_SMILES = mg.sample(50)
TRAIN_SMILES = mg.sample(500)
TEST_SMILES = mg.sample(20)
TARGET_SMILES = mg.sample(20)

MetricEngine = GetMetrics(
    n_jobs=1,
    device='cpu',
    batch_size=512,
    test=TEST_SMILES,
    train=TRAIN_SMILES,
    target=TARGET_SMILES,
)
metrics = MetricEngine.calculate(
    GEN_SMILES,
    calc_valid=True,
    calc_unique=True,
    unique_k=10000,
    se_k=1000,
    sp_k=1000,
    properties=True,
)

```

The output metrics is a dictionary of the following.

    {'#': 50,
    'Validity': 1.0,
    '# valid': 50,
    'Uniqueness': 1.0,
    '# valid & unique': 50,
    'Novelty': 1.0,
    '# novel': 50,
    'IntDiv1': 0.8529291765213013,
    'IntDiv2': 0.8043478185798607,
    'SEDiv': 1.0, 
    'SPDiv': 0.9940201680898508,
    '# scaffolds': 46,
    'ScaffDiv': 0.8308415503042181,
    'ScaffUniqueness': 0.92,
    'FG': 0.2798165137614679,
    'RS': 0.4205607476635514,
    'Filters': 1.0,
    'Purchasable_ZINC20': 0.22,
    'Novelty_test': 1.0,
    'AnSim_test': 0.02,
    'AnCov_test': 0.05,
    'FG_test': 0.8885368776358331,
    'RS_test': 0.9424242286884468,
    'SNN_test': 0.23820277586579322,
    'Frag_test': 0.8192428327241887,
    'Scaf_test': 0.03277367626722305,
    'OutlierBits_test': 0.4735731079340431,
    'logP_test': 0.4607902,
    'NP_test': 0.2155383463278892,
    'SA_test': 0.14422437049300585,
    'QED_test': 0.032100843138596576,
    'Weight_test': 4.263870000000003,
    'Novelty_target': 1.0,
    'AnSim_target': 0.0,
    'AnCov_target': 0.0,
    'FG_target': 0.856314240721427,
    'RS_target': 0.9088194470070764,
    'SNN_target': 0.21473079577088355,
    'Frag_target': 0.6379713633182363,
    'Scaf_target': 0.0,
    'OutlierBits_target': 0.46795784788767686,
    'logP_target': 0.3103534,
    'NP_target': 0.27597897344091976,
    'SA_target': 0.3122152921916008,
    'QED_target': 0.022669396787612487,
    'Weight_target': 9.505290000000002}