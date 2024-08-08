Dimorphite-DL 1.2.4
===================

What is it?
-----------

Dimorphite-DL adds hydrogen atoms to molecular representations, as appropriate
for a user-specified pH range. It is a fast, accurate, accessible, and modular
open-source program for enumerating small-molecule ionization states.

Users can provide SMILES strings from the command line or via an .smi file.

Citation
--------

If you use Dimorphite-DL in your research, please cite:

Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
open-source program for enumerating the ionization states of drug-like small
molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.

Licensing
---------

Dimorphite-DL is released under the Apache 2.0 license. See LICENCE.txt for
details.

Usage
-----

```
usage: dimorphite_dl.py [-h] [--min_ph MIN] [--max_ph MAX]
                        [--pka_precision PRE] [--smiles SMI]
                        [--smiles_file FILE] [--output_file FILE]
                        [--label_states] [--test]

Dimorphite 1.2.4: Creates models of appropriately protonated small moleucles.
Apache 2.0 License. Copyright 2020 Jacob D. Durrant.

optional arguments:
  -h, --help           show this help message and exit
  --min_ph MIN         minimum pH to consider (default: 6.4)
  --max_ph MAX         maximum pH to consider (default: 8.4)
  --pka_precision PRE  pKa precision factor (number of standard devations,
                       default: 1.0)
  --smiles SMI         SMILES string to protonate
  --smiles_file FILE   file that contains SMILES strings to protonate
  --output_file FILE   output file to write protonated SMILES (optional)
  --label_states       label protonated SMILES with target state (i.e.,
                       "DEPROTONATED", "PROTONATED", or "BOTH").
  --test               run unit tests (for debugging)
```

The default pH range is 6.4 to 8.4, considered biologically relevant pH.

Examples
--------

```
  python dimorphite_dl.py --smiles_file sample_molecules.smi
  python dimorphite_dl.py --smiles "CCC(=O)O" --min_ph -3.0 --max_ph -2.0
  python dimorphite_dl.py --smiles "CCCN" --min_ph -3.0 --max_ph -2.0 --output_file output.smi
  python dimorphite_dl.py --smiles_file sample_molecules.smi --pka_precision 2.0 --label_states
  python dimorphite_dl.py --test
```

Advanced Usage
--------------

It is also possible to access Dimorphite-DL from another Python script, rather
than from the command line. Here's an example:

```python
from rdkit import Chem
import dimorphite_dl

# Using the dimorphite_dl.run() function, you can run Dimorphite-DL exactly as
# you would from the command line. Here's an example:
dimorphite_dl.run(
   smiles="CCCN",
   min_ph=-3.0,
   max_ph=-2.0,
   output_file="output.smi"
)
print("Output of first test saved to output.smi...")

# Using the dimorphite_dl.run_with_mol_list() function, you can also pass a
# list of RDKit Mol objects. The first argument is always the list.
smiles = ["C[C@](F)(Br)CC(O)=O", "CCCCCN"]
mols = [Chem.MolFromSmiles(s) for s in smiles]
for i, mol in enumerate(mols):
    mol.SetProp("msg","Orig SMILES: " + smiles[i])

protonated_mols = dimorphite_dl.run_with_mol_list(
    mols,
    min_ph=5.0,
    max_ph=9.0,
)
print([Chem.MolToSmiles(m) for m in protonated_mols])

# Note that properties are preserved.
print([m.GetProp("msg") for m in protonated_mols])
```

Caveats
-------

Dimorphite-DL deprotonates indoles and pyrroles around pH 14.5. But these
substructures can also be protonated around pH -3.5. Dimorphite does not
perform the protonation.

Authors and Contacts
--------------------

See the `CONTRIBUTORS.md` file for a full list of contributors. Please contact
Jacob Durrant (durrantj@pitt.edu) with any questions.
