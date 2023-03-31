# Creating custom scoring functions

In order to define a custom scoring function to work with MolScore a module with a class can be created and put inside the scoring functions folder. The class must adhere to the following format:

``` python
class SomeScoringFunction:
    """Description"""  # Description should be at the class level, this is passed to the config GUI
    return_metrics = <list>  # Name of metrics returned so that they can be selected in the config GUI
    
    def __init__(self, prefix: str, **kwargs):
        # Typing and default values are passed to the config GUI  
        # PyCharm style docstring should be used for parameters only (example below), these are passed to the config GUI
        # Additional choices can be specified in square brackets, for example, [Choice 1, Choice 2, Choice 3]. This will result in a dropdown list in the config GUI. Hence, avoid the use of square brackets otherwise
        """
        :param prefix: Description
        """
        self.prefix = prefix.strip().replace(' ', '_')  # Prefix to seperate multiple uses of the same class
    
    def __call__(self, smiles: list, file_names, directory, **kwargs) -> list[dict]:
        return results  
        # Results should be a list of dictionaries. Each dictionary corresponds to an input SMILES and should 'smiles' key. Every other key should be '<prefix>_<return_metric>'
        # For example,
        # [{'smiles': 'c1ccccc1', 'prefix_docking_score': -7.8},
        #  {'smiles': 'c1ccccc1C(=O)O', 'prefix_docking_score': -9.0]
```

The class should be able to handle invalid smiles/molecules and/or expected errors. In this case, the corresponding dictionary should set all values to 0.0 (except for 'smiles').

To integrate the new scoring function, just add it to the `__init__.py` file as such.

```python
from molscore.scoring_functions.some_scoring_function import SomeScoringFunction

all_scoring_functions=[
    SomeScoringFunction
]
```