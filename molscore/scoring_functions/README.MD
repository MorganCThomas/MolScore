# Defining scoring functions

In order to define your own scoring function to work with MolScore it must adhere to the following:
- Be a python class

- Parameters definable in the config file are designed to be passed to \__init\__
    - \__init\__ must have the following parameters:
        - prefix: str
        
- The \__call\__ method must calculate the score for a list of smiles
    - \__call\__ must have the following parameters:
        - smiles: A unique list of smiles
        - **kwargs:
    - \__call\__ must return the following:
        - return: A dictionary of the results including 'smiles' e.g.
        {'smiles': 'c1ccccc1', '\<prefix>_docking_score': -10.4}. 
        It must populate smiles that error'd with 0.0 as metric vlaues. 