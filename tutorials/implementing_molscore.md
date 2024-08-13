# Implementing MolScore into a generative model

If you wish to follow along with the code examples here, there is a `MockGenerator` class to mimic the sampling of SMILES.
```python
from molscore import MockGenerator
mg = MockGenerator()
SMILES = mg.sample(50)
```

Contents
---
1. [Running a single objective](#single-mode)
2. Running a benchmark
   - [Using a preset benchmark](#using-a-preset-benchmark)
   - [Defining a custom benchmark](#defining-a-custom-benchmark)
   - [Controlling benchmark objectives](#controlling-objectives-in-your-benchmark)
3. [Running curriculum learning](#curriculum-mode)
4. [Using a replay buffer](#using-a-replay-buffer)


## Single mode

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

Moreover, single tasks can be run by name if they already exist as a preset in MolScore configs (such as [Benchmark tasks](#using-a-preset-benchmark)). For this, you must specify the group/category and name of the config file omitting the .json. suffix. If the .json suffix is specified, a path to a file is assumed. See example below,
```python
from molscore import MolScore
ms = MolScore(model_name='test', task_config='GuacaMol:Albuterol_similarity', budget=10000)
while not ms.finished:
    scores = ms.score(SMILES)
```

A list of available configuration files can found here by running the following code snippet,
```python
from molscore import MolScore
named_tasks = {k:[p.name.strip(".json") for p in v.glob("*.json")] for k, v in MolScore.presets.items()}
print(named_tasks)
```
The output will look something like this ...
```json
{"Misc": ["QED", "valid", "valid_unique"],
 "GuacaMol": ["C11H24",
  "C9H10N2O2PF2Cl",
  "Fexofenadine_MPO",
  "Mestranol_similarity",
  "Ranolazine_MPO",
  "Albuterol_similarity",
  ...,
 ],
 ...,
}
```

## Benchmark mode
### Using a preset benchmark

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

### Defining a custom benchmark
Want to run your own benchmark, sure why not!

Simply write the config files that define each individual objective, put them in a directory, and MolScore will run all configs in that directory. The only difference here supplying the path to `custom_benchmark` instead.
```python
from molscore import MolScoreBenchmark
msb = MolScoreBenchmark(model_name='test', output_dir='./', custom_benchmark='path_to_dir', budget=10000)
```

### Controlling objectives in your benchmark
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

## Curriculum mode

Curriculum learning can be viewed as a sequential list of objectives, where you move on to the next objective once some criteria is reached on the current objective (see later).

This is used with a similar API to MolScoreBenchmark, i.e., we specify a directory with a list of configuration files (objectives), but this time, we need to number the files in the order they should be run, or how else would we know? For example,

```bash
curriculum_dir/1_valid.json
curriculum_dir/2_qed.json
curriculum_dir/3_fep.json
```

Once that is defined, we can implement Curriculum learning,

```python
from molscore import MolScoreCurriculum

MolScoreCurriculum.presets.keys()
> dict_keys([]) # Well currently nothing, but a placeholder for some presets / benchmark curriculums

MSC = MolScoreCurriculum(
        model_name="test",
        output_dir=output_directory,
        custom_benchmark=benchmark,
        budget=None,
        termination_threshold=None,
        termination_patience=None,
    )
while not MSC.finished:
    MSC.score(smiles)
```

Ah, but what are the criteria... well we can utilise 3 parameters to control this in order of precedence, specifically:

- **budget** How many molecules should be run for each objective. 
  - NOTE: This takes precedence, if it is specified, the objective will be terminated upon budget reached.
- **termination_threshold** Once we reach this threshold of our overall desirability score / reward, move on to the next objective.
  - NOTE: This is only evaluated if no budget has been specified, or the budget hasn't been reached yet i.e., whichever comes first.
  - RECOMMENDATION: It is probably best not to use this alone. With no budget, what if our threshold is never observed? Therefore, you should probably set a very high budget just incase.
- **termination_patience** How patient are we before moving on to the next objective, this value is the number of iterations (calls to `.score()`) we wait either after reaching a specified termination_threshold.
  - NOTE: If no termination_threshold is specified, termination_patience will conduct early-stopping based on the specified number of iterations where no improvement in the mean is observed.

Where do we specify our new curriculum learning parameters?
- Highest level in our JSON configuration file. 
- `MolScoreCurriculum` and this will override anything specified in the configuration file and apply it to all objectives.

See the example below conducted on batches of 10 randomly sampled SMILES (i.e., no optimization is happening here).

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/cl_example.png?raw=True)

## Using a replay buffer

MolScore can optionally maintain a replay buffer of the top-n compounds to be sampled and used for more efficient learning. This can be used in any of the aforementioned modes i.e., Single, Benchmark and Curriculum.

To control this, we have two parameters:

- **replay_size** How many molecules to store in the replay buffer.
- **replay_purge** If True, only allow molecules that pass the diversity filter used into the replay buffer, a.k.a [augmented-memory](https://pubs.acs.org/doi/10.1021/jacsau.4c00066). THis helps to prevent mode collapse if experience replay is used excessively. 
  - NOTE: If no diversity filter is being used, the buffer cannot be purged. This will throw a warning but continue to function without purging non-diverse molecules.

These are passed during any mode instantion e.g.,
```python
ms = MolScore(
    model_name='test',
    task_config='molscore/configs/QED.json',
    replay_size=100, # MolScore, MolScoreBenchmark, or MolScoreCurriculum
    replay_purge=True # MolScore, MolScoreBenchmark, or MolScoreCurriculum
    )
```

Then, at any time, you can use the `replay` method to sample `n` molecules from the replay buffer.
```python
smiles, scores = ms.replay(n=20, augment=False)
```
**Note**: Passing `augment=True` will provide a randomized version of the smiles for [data augmentation](https://arxiv.org/abs/2210.12458).

**How does this work with MolScoreBenchmark?** Well this simple creates a buffer for every task.  
**How does this work with MolScoreCurriculum?** This creates a shared buffer accross tasks that will be updated, to reset the buffer between tasks set `reset_replay_buffer=True` during instantion.





