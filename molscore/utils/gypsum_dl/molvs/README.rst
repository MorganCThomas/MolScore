MolVS: Molecule Validation and Standardization
==============================================

.. image:: https://img.shields.io/pypi/v/MolVS.svg?style=flat-square
    :alt: PyPI package
    :target: https://pypi.python.org/pypi/MolVS

.. image:: https://img.shields.io/conda/vn/conda-forge/molvs.svg?style=flat-square
    :alt: Conda package
    :target: https://anaconda.org/conda-forge/molvs

.. image:: https://img.shields.io/github/license/mcs07/MolVS.svg?style=flat-square
    :alt: MIT license
    :target: https://github.com/mcs07/MolVS/blob/master/LICENSE

.. image:: https://img.shields.io/azure-devops/build/mcs07/MolVS/1.svg?style=flat-square
    :alt: Azure DevOps tests
    :target: https://dev.azure.com/mcs07/MolVS/_build?definitionId=1

**MolVS** is a molecule validation and standardization tool, written in Python using the `RDKit chemistry framework`_.

Building a collection of chemical structures from different sources can be difficult due to differing representations,
drawing conventions and mistakes. MolVS can standardize chemical structures to improve data quality, help with
de-duplication and identify relationships between molecules.

There are sensible defaults that make it easy to get started::

    >>> from molvs import standardize_smiles
    >>> standardize_smiles('[Na]OC(=O)c1ccc(C[S+2]([O-])([O-]))cc1')
    '[Na+].O=C([O-])c1ccc(CS(=O)=O)cc1'

Installation
------------

To install MolVS with Anaconda Python, simply run::

    conda install -c conda-forge molvs

Alternatively, try one of the other `installation options`_.

Documentation
-------------

Full documentation is available at https://molvs.readthedocs.io.

Contribute
----------

-  Feature ideas and bug reports are welcome on the `Issue Tracker`_.
-  Fork the `source code`_ on GitHub, make changes and send a pull request.

License
-------

MolVS is licensed under the `MIT license`_.

Similar projects
----------------

There are a number of projects with similar goals that take differing approaches:

- `Francis Atkinson's Standardiser`_
- `RSC Chemistry Validation and Standardization Platform (CVSP)`_
- `PubChem Standardization Service`_
- `Tripod Structure standardizer`_
- `FDA Substance Registration System Standard Operating Procedure`_
- `ChemAxon Structure Standardizer`_


.. _`RDKit chemistry framework`: http://www.rdkit.org
.. _`installation options`: http://molvs.readthedocs.io/en/latest/guide/install.html
.. _`source code`: https://github.com/mcs07/MolVS
.. _`Issue Tracker`: https://github.com/mcs07/MolVS/issues
.. _`MIT license`: https://github.com/mcs07/MolVS/blob/master/LICENSE
.. _`Francis Atkinson's Standardiser`: https://wwwdev.ebi.ac.uk/chembl/extra/francis/standardiser/
.. _`RSC Chemistry Validation and Standardization Platform (CVSP)`: http://cvsp.chemspider.com
.. _`PubChem Standardization Service`: https://pubchem.ncbi.nlm.nih.gov/standardize/standardize.cgi
.. _`Tripod Structure standardizer`: https://tripod.nih.gov/?p=61
.. _`FDA Substance Registration System Standard Operating Procedure`: http://www.fda.gov/downloads/ForIndustry/DataStandards/SubstanceRegistrationSystem-UniqueIngredientIdentifierUNII/ucm127743.pdf
.. _`ChemAxon Structure Standardizer`: http://www.chemaxon.com/products/standardizer/
