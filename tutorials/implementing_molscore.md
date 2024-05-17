# Implementing MolScore into a generative model

If you wish to follow along with the code examples here, there is a `MockGenerator` class to mimic the sampling of SMILES.
```python
from molscore import MockGenerator
mg = MockGenerator()
SMILES = mg.sample(50)
```

## Running a single objective

Implementing `molscore` is as simple as importing it, instantiating it (pointing to the configuration file) and then scoring molecules. This should easily fit into most generative model pipelines. Other MolScore parameters include `output_dir` to override any specified in the `task_config`.

```python
from molscore import MolScore

ms = MolScore(model_name='test', task_config='molscore/configs/QED.json')
scores = ms.score(SMILES)
```

Alternatively, a `budget` can be set to specify the maximum number of molecules to score, after the budget is reached, the attribute `finished` will be set to `True` which can be used to decide when to exit an optimization loop. For example,
```python
from molscore import MolScore
ms = MolScore(model_name='test', task_config='molscore/configs/QED.json', budget=10000)
while not ms.finished:
    scores = ms.score(SMILES)
```

## Running a preset benchmark

A benchmark mode is also available, that can be used to iterate over objectives and automatically computes some metrics upon finishing. The following pre-defined benchmarks come packaged with MolScore including:
- [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839)
- [GuacaMol_Scaffold](https://arxiv.org/pdf/2103.03864.pdf)
- [MolOpt](https://arxiv.org/abs/2206.12411)
- [5HT2A_PhysChem](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda)
- [5HT2A_Selectivity](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda)
- [5HT2A_Docking](https://chemrxiv.org/engage/chemrxiv/article-details/65e63a4de9ebbb4db9e63fda)
- [LibINVENT_Exp1](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00469)
- [LinkINVENT_Exp3](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00115b)

These can also be found in the `presets` attribute of `MolScoreBenchmark`.

Here is an example to run a preset benchmark.
```python
from molscore import MolScoreBenchmark

MolScoreBenchmark.presets.keys()
> dict_keys(['GuacaMol', 'GuacaMol_Scaffold', 'MolOpt', '5HT2A_PhysChem', '5HT2A_Selectivity', '5HT2A_Docking', 'LibINVENT_Exp1', 'LinkINVENT_Exp3'])

msb = MolScoreBenchmark(model_name='test', output_dir='./', benchmark='GuacaMol', budget=10000)
for task in msb:
    while not task.finished:
        scores = task.score(SMILES)
```

## Running a custom benchmark
Want to run your own benchmark, sure why not!

Simply write the config files that define each individual objective, put them in a directory, and MolScore will run all configs in that directory. The only difference here supplying the path to `custom_benchmark` instead.
```python
from molscore import MolScoreBenchmark
msb = MolScoreBenchmark(model_name='test', output_dir='./', custom_benchmark='path_to_dir', budget=10000)
```

## Controlling objectives in your benchmark
Want more control what to (or what not to) run? Sure, use the `include` and `exclude` parameters. For example, I can run GuacaMol, and my own benchmark, but exclude `albuterol_similarity`.
```python
from molscore import MolScoreBenchmark
msb = MolScoreBenchmark(
    model_name='test',
    benchmark='GuacaMol',
    custom_benchmark='path_to_dir',
    exclude=['albuterol_similarity']
    budget=10000)
```

Or if I only want to run specific objectives, I can use include.
```python
from molscore import MolScoreBenchmark
msb = MolScoreBenchmark(
    model_name='test',
    benchmark='MolOpt',
    include=['albuterol_similarity', 'qed']
    budget=10000)
```


