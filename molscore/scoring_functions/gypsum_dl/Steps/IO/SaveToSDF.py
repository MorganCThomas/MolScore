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
Saves output files to SDF.
"""

import __future__
import os

import molscore.scoring_functions.gypsum_dl.Utils as Utils

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


def save_to_sdf(contnrs, params, separate_output_files, output_folder):
    """Saves the 3D models to the disk as an SDF file.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param params: The parameters.
    :type params: dict
    :param separate_output_files: Whether save each molecule to a different
       file.
    :type separate_output_files: bool
    :param output_folder: The output folder.
    :type output_folder: str
    """

    # Save an empty molecule with the parameters.
    if separate_output_files == False:
        w = Chem.SDWriter(output_folder + os.sep + "gypsum_dl_success.sdf")
    else:
        w = Chem.SDWriter(output_folder + os.sep + "gypsum_dl_params.sdf")

    m = Chem.Mol()
    m.SetProp("_Name", "EMPTY MOLECULE DESCRIBING GYPSUM-DL PARAMETERS")
    for param in params:
        m.SetProp(param, str(params[param]))
    w.write(m)

    if separate_output_files == True:
        w.flush()
        w.close()

    # Also save the file or files containing the output molecules.
    Utils.log("Saving molecules associated with...")
    for i, contnr in enumerate(contnrs):
        # Add the container properties to the rdkit_mol object so they get
        # written to the SDF file.
        contnr.add_container_properties()

        # Let the user know which molecule you're on.
        Utils.log("\t" + contnr.orig_smi)

        # Save the file(s).
        if separate_output_files == True:
            # sdf_file = "{}{}__{}.pdb".format(output_folder + os.sep, slug(name), conformer_counter)
            sdf_file = "{}{}__input{}.sdf".format(
                output_folder + os.sep,
                Utils.slug(contnr.name),
                contnr.contnr_idx_orig + 1,
            )
            w = Chem.SDWriter(sdf_file)
            # w = Chem.SDWriter(output_folder + os.sep + "output." + str(i + 1) + ".sdf")

        for m in contnr.mols:
            m.load_conformers_into_rdkit_mol()
            w.write(m.rdkit_mol)

        if separate_output_files == True:
            w.flush()
            w.close()

    if separate_output_files == False:
        w.flush()
        w.close()
