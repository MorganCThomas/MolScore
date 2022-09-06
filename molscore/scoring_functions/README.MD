# Defining scoring functions

In order to define your own scoring function to work with MolScore it must adhere to the following format:

``` python
class SomeClass:
    SomeClass.return_metrics = <list>  # List of metrics returned
    
    def __init__(self, prefix: str, **kwargs):
        self.prefix = prefix.strip().replace(' ', '_')
    
    def __call__(self, smiles: list, **kwargs) -> list[dict]:
        return results  
        # results should have a 'smiles' key and every other key should be '<prefix>_<return_metric>'
        # e.g. [{'smiles': 'c1ccccc1', 'test_docking_score': -7.8},
        #       {'smiles': 'c1ccccc1C(=O)O', 'test_docking_score': -9.0]
```

Furthermore, the class should handle invalid smiles/molecules and/or errors.
In this case, the corresponding dictionary should set all values to 0.0 (except for 'smiles').

To integrate the new scoring function, just add it to the `__init__.py` file as such.

```python
from molscore.scoring_functions.some_new_function import SomeNewFunction

all_scoring_functions=[
    SomeNewFunction
]
```