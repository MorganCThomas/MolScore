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
Desalts the input SMILES strings. If an input SMILES string contains to
molecule, keep the larger one.
"""

import __future__

import gypsum_dl.Parallelizer as Parallelizer
import gypsum_dl.Utils as Utils
import gypsum_dl.ChemUtils as ChemUtils
import gypsum_dl.MyMol as MyMol

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


def desalt_orig_smi(
    contnrs, num_procs, job_manager, parallelizer_obj, durrant_lab_filters=False
):
    """If an input molecule has multiple unconnected fragments, this removes
       all but the largest fragment.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param num_procs: The number of processors to use.
    :type num_procs: int
    :param job_manager: The multiprocess mode.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    """

    Utils.log("Desalting all molecules (i.e., keeping only largest fragment).")

    # Desalt each of the molecule containers. This step is very fast, so let's
    # just run it on a single processor always.
    tmp = [desalter(x) for x in contnrs]

    # Go through each contnr and update the orig_smi_deslt. If we update it,
    # also add a note in the genealogy record.
    tmp = Parallelizer.strip_none(tmp)
    for idx in range(0, len(tmp)):
        desalt_mol = tmp[idx]
        # idx = desalt_mol.contnr_idx
        cont = contnrs[idx]

        if contnrs[idx].orig_smi != desalt_mol.orig_smi:
            desalt_mol.genealogy.append(desalt_mol.orig_smi_deslt + " (desalted)")
            cont.update_orig_smi(desalt_mol.orig_smi_deslt)

        cont.add_mol(desalt_mol)


def desalter(contnr):
    """Desalts molecules in a molecule container.

    :param contnr: The molecule container.
    :type contnr: MolContainer.MolContainer
    :return: A molecule object.
    :rtype: MyMol.MyMol
    """

    # Split it into fragments
    frags = contnr.get_frags_of_orig_smi()

    if len(frags) == 1:
        # It's only got one fragment, so default assumption that
        # orig_smi = orig_smi_deslt is correct.
        return contnr.mol_orig_frm_inp_smi
    else:
        Utils.log(
            "\tMultiple fragments found in "
            + contnr.orig_smi
            + " ("
            + contnr.name
            + ")"
        )

        # Find the biggest fragment
        num_heavy_atoms = []
        num_heavy_atoms_to_frag = {}

        for i, f in enumerate(frags):
            num = f.GetNumHeavyAtoms()
            num_heavy_atoms.append(num)
            num_heavy_atoms_to_frag[num] = f

        max_num = max(num_heavy_atoms)
        biggest_frag = num_heavy_atoms_to_frag[max_num]

        # Return info about that biggest fragment.
        new_mol = MyMol.MyMol(biggest_frag)
        new_mol.contnr_idx = contnr.contnr_idx
        new_mol.name = contnr.name
        new_mol.genealogy = contnr.mol_orig_frm_inp_smi.genealogy
        new_mol.make_mol_frm_smiles_sanitze()  # Need to update the mol.
        return new_mol
