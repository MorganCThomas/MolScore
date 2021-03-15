# Copyright 2018 Jacob D. Durrant
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains the function for saving the output to PDB files.
"""

import __future__

import glob
import sys
import os
from os.path import basename

from gypsum_dl import Utils
import rdkit
import rdkit.Chem as Chem

# Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "gypsum_dl"))


def convert_sdfs_to_PDBs(contnrs, output_folder):
    """This will convert every conformer into a PDB file, which is saved in
       the output_folder.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param output_folder: The name of the output folder.
    :type output_folder: str
    """

    # Go through each container.
    for contnr in contnrs:
        contnr.add_container_properties()

        # Get the molecule name and associated variants.
        name = contnr.name
        mols = contnr.mols

        # Got through the variants.
        for i, m in enumerate(mols):
            pdb_file = "{}{}__input{}__variant{}.pdb".format(
                output_folder + os.sep,
                Utils.slug(name),
                contnr.contnr_idx_orig + 1,
                i + 1,
            )

            # Get the conformers into the rdkit_mol object.
            m.load_conformers_into_rdkit_mol()
            mol = m.rdkit_mol
            if mol == None:
                continue
            else:
                # Write conformers to a PDB file.
                Chem.MolToPDBFile(mol, pdb_file, flavor=32)

                # Add header to PDB file with original SMILES and final SMILES
                printout = "REMARK Original SMILES string: {}\nREMARK Final SMILES string: {}\n".format(
                    m.orig_smi, m.standardize_smiles()
                )
                with open(pdb_file) as f:
                    printout = printout + f.read()
                with open(pdb_file, "w") as f:
                    f.write(printout)
                printout = ""
