# Fréchet ChemNet Distance on PyTorch

[![Build Status](https://travis-ci.com/insilicomedicine/fcd_torch.svg?branch=master)](https://travis-ci.com/insilicomedicine/fcd_torch) [![PyPI version](https://badge.fury.io/py/fcd-torch.svg)](https://badge.fury.io/py/fcd-torch)

PyTorch implementation of [Fréchet ChemNet Distance](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234) ported from the original repository https://github.com/bioinf-jku/FCD. The ported model produces the same outputs as the original Keras implementation and can be used for reproducible research. The PyTorch model of ChemNet weights tenfold less, resulting in faster loading.


Other features:
* You can precalculate mean and sigma for further usage, useful if you use the statistics from the same dataset multiple times
* Supports calculation on GPU and selection of GPU device number
* Multithreaded SMILES parsing


## Installation
First, install [RDKit](https://www.rdkit.org/docs/Install.html): `conda install -yq -c rdkit rdkit` and then install `fcd_torch` from pip (`pip install fcd_torch`), or directly from the source:
```{bash}
git clone https://github.com/insilicomedicine/fcd_torch.git
cd fcd_torch
python setup.py install
```

## Usage

Import the module `from fcd_torch import FCD`. You can run calculation directly or precalculate statistics to reuse them on the test set (see example below). If you run FCD on GPU, the GPU memory will be allocated only during calculation of FCD.

```python
# Example 1:
    fcd = FCD(device='cuda:0', n_jobs=8)
    smiles_list1 = ['COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1', 'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1']
    smiles_list2 = ['Oc1ccccc1-c1cccc2cnccc12', 'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1']
    fcd(smiles_list1, smiles_list2)
```

```python
# Example 2:
    fcd = FCD(device='cuda:0', n_jobs=8)
    smiles_list1 = ['COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1', 'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1']
    smiles_list2 = ['Oc1ccccc1-c1cccc2cnccc12', 'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1']
    pgen = fcd.precalc(smiles_list2)
    fcd(smiles_list1, pgen=pgen)
```

For the constructor, you can pass the device as `device='cpu'` for CPU and `device='cuda:n'` for GPU, where `n` is the GPU device number. `n_jobs` parameter specifies the number of threads for parsing SMILES. You can also vary the `batch_size` parameter. Call parameters for FCD are `fcd(ref=None, gen=None, pref=None, pgen=None)`, where you should specify either `ref` (SMILES list), or `pref` (precalculated statistics), and the same for `gen` and `pgen`.
