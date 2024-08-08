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
A module for generating alternate chiralities.
"""

import __future__

import copy
import itertools
import random

import molscore.scoring_functions.gypsum_dl.Parallelizer as Parallelizer
import molscore.scoring_functions.gypsum_dl.Utils as Utils
import molscore.scoring_functions.gypsum_dl.ChemUtils as ChemUtils
import molscore.scoring_functions.gypsum_dl.MyMol as MyMol

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


def enumerate_chiral_molecules(
    contnrs,
    max_variants_per_compound,
    thoroughness,
    num_procs,
    job_manager,
    parallelizer_obj,
):
    """Enumerates all possible enantiomers of a molecule. If the chirality of
       an atom is given, that chiral center is not varied. Only the chirality
       of unspecified chiral centers is varied.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param max_variants_per_compound: To control the combinatorial explosion,
       only this number of variants (molecules) will be advanced to the next
       step.
    :type max_variants_per_compound: int
    :param thoroughness: How many molecules to generate per variant (molecule)
       retained, for evaluation. For example, perhaps you want to advance five
       molecules (max_variants_per_compound = 5). You could just generate five
       and advance them all. Or you could generate ten and advance the best
       five (so thoroughness = 2). Using thoroughness > 1 increases the
       computational expense, but it also increases the chances of finding good
       molecules.
    :type thoroughness: int
    :param num_procs: The number of processors to use.
    :type num_procs: int
    :param job_manager: The multiprocess mode.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    """

    # No point in continuing none requested.
    if max_variants_per_compound == 0:
        return

    Utils.log("Enumerating all possible enantiomers for all molecules...")

    # Group the molecules so you can feed them to parallelizer.
    params = []
    for contnr in contnrs:
        for mol in contnr.mols:
            params.append(tuple([mol, thoroughness, max_variants_per_compound]))
    params = tuple(params)

    # Run it through the parallelizer.
    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(params, parallel_get_chiral, num_procs, job_manager)
    else:
        for i in params:
            tmp.append(parallel_get_chiral(i[0], i[1], i[2]))

    # Remove Nones (failed molecules)
    clean = Parallelizer.strip_none(tmp)

    # Flatten the data into a single list.
    flat = Parallelizer.flatten_list(clean)

    # Get the indexes of the ones that failed to generate.
    contnr_idxs_of_failed = Utils.fnd_contnrs_not_represntd(contnrs, flat)

    # Go through the missing ones and throw a message.
    for miss_indx in contnr_idxs_of_failed:
        Utils.log(
            "\tCould not generate valid enantiomers for "
            + contnrs[miss_indx].orig_smi
            + " ("
            + contnrs[miss_indx].name
            + "), so using existing "
            + "(unprocessed) structures."
        )
        for mol in contnrs[miss_indx].mols:
            mol.genealogy.append("(WARNING: Unable to generate enantiomers)")
            clean.append(mol)

    # Keep only the top few compound variants in each container, to prevent a
    # combinatorial explosion.
    ChemUtils.bst_for_each_contnr_no_opt(
        contnrs, flat, max_variants_per_compound, thoroughness
    )


def parallel_get_chiral(mol, max_variants_per_compound, thoroughness):
    """A parallelizable function for enumerating chiralities.

    :param mol: The input molecule.
    :type mol: MyMol.MyMol
    :param max_variants_per_compound: To control the combinatorial explosion,
       only this number of variants (molecules) will be advanced to the next
       step.
    :type max_variants_per_compound: int
    :param thoroughness: How many molecules to generate per variant (molecule)
       retained, for evaluation. For example, perhaps you want to advance five
       molecules (max_variants_per_compound = 5). You could just generate five
       and advance them all. Or you could generate ten and advance the best
       five (so thoroughness = 2). Using thoroughness > 1 increases the
       computational expense, but it also increases the chances of finding good
       molecules.
    :type thoroughness: int
    :return: A list of MyMol.MyMol objects.
    :rtype: list
    """

    # Get all chiral centers that aren't assigned explicitly in the input
    # molecules.
    unasignd = [p[0] for p in mol.chiral_cntrs_w_unasignd() if p[1] == "?"]
    num = len(unasignd)

    # Get all possible chiral assignments. If the chirality is specified,
    # retain it.
    results = []
    if num == 0:
        # There are no unspecified chiral centers, so just keep existing.
        results.append(mol)
        return results
    elif num == 1:
        # There's only one chiral center.
        options = ["R", "S"]
    else:
        # There are multiple chiral centers.
        starting = [["R"], ["S"]]
        options = [["R"], ["S"]]
        for i in range(num - 1):
            if len(options) > thoroughness * max_variants_per_compound:
                # Unfortunately, this section lends itself to a combinatorial
                # explosion if there are many chiral centers. Necessary to
                # control that or it can become problematic. So truncate early
                # if you already have a enough (so some will unfortunately
                # never be evaluated).
                break
            options = list(itertools.product(options, starting))
            options = [list(itertools.chain(c[0], c[1])) for c in options]

    # Let the user know the number of chiral centers.
    Utils.log(
        "\t"
        + mol.smiles(True)
        + " ("
        + mol.name
        + ") has "
        # + str(len(options))
        + str(2 ** num)
        + " enantiomers when chiral centers with "
        + "no specified chirality are systematically varied."
    )

    # Randomly select a few of the chiral combinations to examine. This is to
    # reduce the potential combinatorial explosion.
    num_to_keep_initially = thoroughness * max_variants_per_compound
    options = Utils.random_sample(options, num_to_keep_initially, "")

    # Go through the chirality combinations and make a molecule with that
    # chirality.
    for option in options:
        # Copy the initial rdkit molecule.
        a_rd_mol = copy.copy(mol.rdkit_mol)

        # Set its chirality.
        for idx, chiral in zip(unasignd, option):
            if chiral == "R":
                a_rd_mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                )
            elif chiral == "S":
                a_rd_mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
                )

        # Make a new MyMol.MyMol object from that rdkit molecule.
        new_mol = MyMol.MyMol(a_rd_mol)

        # Add the new molecule to the list of results, if it does not have a
        # bizarre substructure.
        if not new_mol.remove_bizarre_substruc():
            new_mol.contnr_idx = mol.contnr_idx
            new_mol.name = mol.name
            new_mol.genealogy = mol.genealogy[:]
            new_mol.genealogy.append(new_mol.smiles(True) + " (chirality)")
            results.append(new_mol)

    # Return the results.
    return results
