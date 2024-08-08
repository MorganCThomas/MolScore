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
The proccess_output definition determines which formats are saved to the
disk (output).
"""

import __future__

from molscore.scoring_functions.gypsum_dl.Steps.IO.SaveToSDF import save_to_sdf
from molscore.scoring_functions.gypsum_dl.Steps.IO.SaveToPDB import convert_sdfs_to_PDBs
from molscore.scoring_functions.gypsum_dl.Steps.IO.Web2DOutput import web_2d_output
from molscore.scoring_functions.gypsum_dl import Utils


def proccess_output(contnrs, params):
    """Proccess the molecular models in preparation for writing them to the
       disk."""

    # Unpack some variables.
    separate_output_files = params["separate_output_files"]
    output_folder = params["output_folder"]

    if params["add_html_output"] == True:
        # Write to an HTML file.
        web_2d_output(contnrs, output_folder)

    # Write to an SDF file.
    save_to_sdf(contnrs, params, separate_output_files, output_folder)

    # Also write to PDB files, if requested.
    if params["add_pdb_output"] == True:
        Utils.log("\nMaking PDB output files\n")
        convert_sdfs_to_PDBs(contnrs, output_folder)
