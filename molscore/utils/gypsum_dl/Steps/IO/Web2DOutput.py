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
Saves the output to an HTML file (2D images only). This is mostly for
debugging.
"""

# import webbrowser
import os
import gypsum_dl.Utils as Utils
import gypsum_dl.ChemUtils as ChemUtils

try:
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.Draw import PrepareMolForDrawing
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


def web_2d_output(contnrs, output_folder):
    """Saves pictures of the models to an HTML file on disk. It can be viewed in
    a browser. Then opens a browser automatically to view them. This is mostly
    for debugging."""

    Utils.log("Saving html image of molecules associated with...")

    # Let's not parallelize it for now. This will rarely be used.
    html_file = output_folder + os.sep + "gypsum_dl_success.html"
    f = open(html_file, "w")
    for contnr in contnrs:
        Utils.log("\t" + contnr.orig_smi)
        for mol in contnr.mols:
            # See
            # http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RemoveHs
            # I think in older versions of rdkit (e.g., 2016.09.2), RemoveHs
            # would remove hydrogens, even if that make double bonds
            # ambiguous. Not so in newer versions (e.g., 2018.03.4). So if
            # your double-bonded nitrogen doesn't have its hydrogen attached,
            # and you're using an older version of rdkit, don't worry about
            # it. The cis/trans info is still there.
            mol2 = Chem.RemoveHs(mol.rdkit_mol)
            # mol2 = mol.rdkit_mol

            mol2 = PrepareMolForDrawing(mol2, addChiralHs=True, wedgeBonds=True)
            rdDepictor.Compute2DCoords(mol2)
            drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
            drawer.DrawMolecule(mol2)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            f.write(
                '<div style="float: left; width:200px; height: 220px;" title="'
                + mol.name
                + '">'
                + '<div style="width: 200px; height: 200px;">'
                + svg.replace("svg:", "")
                + "</div>"
                + '<div style="width: 200px; height: 20px;">'
                + "<small><center>"
                + mol.smiles(True)
                + "</center></small>"
                + "</div>"
                + "</div>"
            )
    f.close()

    # Open the browser to show the file.
    # webbrowser.open("file://" + os.path.abspath(html_file))
