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
This module makes alternate tautomeric states, using MolVS.
"""

import __future__

import random

import gypsum_dl.Parallelizer as Parallelizer
import gypsum_dl.Utils as Utils
import gypsum_dl.ChemUtils as ChemUtils
import gypsum_dl.MyMol as MyMol
import gypsum_dl.MolObjectHandling as MOH

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")

try:
    from gypsum_dl.molvs import tautomer
except:
    Utils.exception("You need to install molvs and its dependencies.")


def make_tauts(
    contnrs,
    max_variants_per_compound,
    thoroughness,
    num_procs,
    job_manager,
    let_tautomers_change_chirality,
    parallelizer_obj,
):
    """Generates tautomers of the molecules. Note that some of the generated
    tautomers are not realistic. If you find a certain improbable
    substructure keeps popping up, add it to the list in the
    `prohibited_substructures` definition found with MyMol.py, in the function
    remove_bizarre_substruc().

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: A list.
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
    :param let_tautomers_change_chirality: Whether to allow tautomers that
      change the total number of chiral centers.
    :type let_tautomers_change_chirality: bool
    :param job_manager: The multithred mode to use.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    """

    # No need to proceed if there are no max variants.
    if max_variants_per_compound == 0:
        return

    Utils.log("Generating tautomers for all molecules...")

    # Create the parameters to feed into the parallelizer object.
    params = []
    for contnr in contnrs:
        for mol_index, mol in enumerate(contnr.mols):
            params.append(tuple([contnr, mol_index, max_variants_per_compound]))
    params = tuple(params)

    # Run the tautomizer through the parallel object.
    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(params, parallel_make_taut, num_procs, job_manager)
    else:
        for i in params:
            tmp.append(parallel_make_taut(i[0], i[1], i[2]))

    # Flatten the resulting list of lists.
    none_data = tmp
    taut_data = Parallelizer.flatten_list(none_data)

    # Remove bad tautomers.
    taut_data = tauts_no_break_arom_rngs(
        contnrs, taut_data, num_procs, job_manager, parallelizer_obj
    )

    if not let_tautomers_change_chirality:
        taut_data = tauts_no_elim_chiral(
            contnrs, taut_data, num_procs, job_manager, parallelizer_obj
        )

    # taut_data = tauts_no_change_hs_to_cs_unless_alpha_to_carbnyl(
    #    contnrs, taut_data, num_procs, job_manager, parallelizer_obj
    # )

    # Keep only the top few compound variants in each container, to prevent a
    # combinatorial explosion.
    ChemUtils.bst_for_each_contnr_no_opt(
        contnrs, taut_data, max_variants_per_compound, thoroughness
    )


def parallel_make_taut(contnr, mol_index, max_variants_per_compound):
    """Makes alternate tautomers for a given molecule container. This is the
       function that gets fed into the parallelizer.

    :param contnr: The molecule container.
    :type contnr: MolContainer.MolContainer
    :param mol_index: The molecule index.
    :type mol_index: int
    :param max_variants_per_compound: To control the combinatorial explosion,
       only this number of variants (molecules) will be advanced to the next
       step.
    :type max_variants_per_compound: int
    :return: A list of MyMol.MyMol objects, containing the alternate
        tautomeric forms.
    :rtype: list
    """

    # Get the MyMol.MyMol within the molecule container corresponding to the
    # given molecule index.
    mol = contnr.mols[mol_index]

    # Create a temporary RDKit mol object, since that's what MolVS works with.
    # TODO: There should be a copy function
    m = MyMol.MyMol(mol.smiles()).rdkit_mol

    # For tautomers to work, you need to not have any explicit hydrogens.
    m = Chem.RemoveHs(m)

    # Make sure it's not None.
    if m is None:
        Utils.log(
            "\tCould not generate tautomers for "
            + contnr.orig_smi
            + ". I'm deleting it."
        )
        return

    # Molecules should be kekulized already, but let's double check that.
    # Because MolVS requires kekulized input.
    Chem.Kekulize(m)
    m = MOH.check_sanitization(m)
    if m is None:
        return None

    # Limit to max_variants_per_compound tauts. Note that another batch could
    # add more, so you'll need to once again trim to this number later. But
    # this could at least help prevent the combinatorial explosion at this
    # stage.
    enum = tautomer.TautomerEnumerator(max_tautomers=max_variants_per_compound)
    tauts_rdkit_mols = enum.enumerate(m)

    # Make all those tautomers into MyMol objects.
    tauts_mols = [MyMol.MyMol(m) for m in tauts_rdkit_mols]

    # Keep only those that have reasonable substructures.
    tauts_mols = [t for t in tauts_mols if t.remove_bizarre_substruc() == False]

    # If there's more than one, let the user know that.
    if len(tauts_mols) > 1:
        Utils.log("\t" + mol.smiles(True) + " has tautomers.")

    # Now collect the final results.
    results = []

    for tm in tauts_mols:
        tm.inherit_contnr_props(contnr)
        tm.genealogy = mol.genealogy[:]
        tm.name = mol.name

        if tm.smiles() != mol.smiles():
            tm.genealogy.append(tm.smiles(True) + " (tautomer)")

        results.append(tm)

    return results


def tauts_no_break_arom_rngs(
    contnrs, taut_data, num_procs, job_manager, parallelizer_obj
):
    """For a given molecule, the number of atomatic rings should never change
       regardless of tautization, ionization, etc. Any taut that breaks
       aromaticity is unlikely to be worth pursuing. So remove it.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: A list.
    :param taut_data: A list of MyMol.MyMol objects.
    :type taut_data: list
    :param num_procs: The number of processors to use.
    :type num_procs: int
    :param job_manager: The multithred mode to use.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    :return: A list of MyMol.MyMol objects, with certain bad ones removed.
    :rtype: list
    """

    # You need to group the taut_data by container to pass it to the
    # paralleizer.
    params = []
    for taut_mol in taut_data:
        for contnr in contnrs:
            if contnr.contnr_idx == taut_mol.contnr_idx:
                container = contnr

        params.append(tuple([taut_mol, container]))
    params = tuple(params)

    # Run it through the parallelizer to remove non-aromatic rings.

    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(
            params, parallel_check_nonarom_rings, num_procs, job_manager
        )
    else:
        for i in params:
            tmp.append(parallel_check_nonarom_rings(i[0], i[1]))

    # Stripping out None values (failed).
    results = Parallelizer.strip_none(tmp)

    return results


def tauts_no_elim_chiral(contnrs, taut_data, num_procs, job_manager, parallelizer_obj):
    """Unfortunately, molvs sees removing chiral specifications as being a
       distinct taut. I imagine there are cases where tautization could
       remove a chiral center, but I think these cases are rare. To compensate
       for the error in other folk's code, let's just require that the number
       of chiral centers remain unchanged with isomerization.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param taut_data: A list of MyMol.MyMol objects.
    :type taut_data: list
    :param num_procs: The number of processors to use.
    :type num_procs: int
    :param job_manager: The multithred mode to use.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    :return: A list of MyMol.MyMol objects, with certain bad ones removed.
    :rtype: list
    """

    # You need to group the taut_data by contnr to pass to paralleizer.
    params = []
    for taut_mol in taut_data:
        taut_mol_idx = int(taut_mol.contnr_idx)

        for contnr in contnrs:
            if contnr.contnr_idx == taut_mol.contnr_idx:
                container = contnr

        params.append(tuple([taut_mol, container]))
    params = tuple(params)

    # Run it through the parallelizer.
    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(
            params, parallel_check_chiral_centers, num_procs, job_manager
        )
    else:
        for i in params:
            tmp.append(parallel_check_chiral_centers(i[0], i[1]))

    # Stripping out None values
    results = [x for x in tmp if x != None]

    return results


def tauts_no_change_hs_to_cs_unless_alpha_to_carbnyl(
    contnrs, taut_data, num_procs, job_manager, parallelizer_obj
):
    """Generally speaking, only carbons that are alpha to a carbonyl are
       sufficiently acidic to participate in tautomer formation. The
       tautomer-generating code you use makes these inappropriate tautomers.
       Remove them here.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param taut_data: A list of MyMol.MyMol objects.
    :type taut_data: list
    :param num_procs: The number of processors to use.
    :type num_procs: int
    :param job_manager: The multithred mode to use.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    :return: A list of MyMol.MyMol objects, with certain bad ones removed.
    :rtype: list
    """

    # Group the taut_data by container to run it through the parallelizer.
    params = []
    for taut_mol in taut_data:
        params.append(tuple([taut_mol, contnrs[taut_mol.contnr_idx]]))
    params = tuple(params)

    # Run it through the parallelizer.
    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(
            params, parallel_check_carbon_hydrogens, num_procs, job_manager
        )
    else:
        for i in params:
            tmp.append(parallel_check_carbon_hydrogens(i[0], i[1]))

    # Strip out the None values.
    results = [x for x in tmp if x != None]

    return results


def parallel_check_nonarom_rings(taut, contnr):
    """A parallelizable helper function that checks that tautomers do not
       break any nonaromatic rings present in the original object.

    :param taut: The tautomer to evaluate.
    :type taut: MyMol.MyMol
    :param contnr: The original molecule container.
    :type contnr: MolContainer.MolContainer
    :return: Either the tautomer or a None object.
    :rtype: MyMol.MyMol | None
    """

    # How many nonaromatic rings in the original smiles?
    num_nonaro_rngs_orig = contnr.num_nonaro_rngs

    # Check if it breaks aromaticity.
    get_idxs_of_nonaro_rng_atms = len(taut.get_idxs_of_nonaro_rng_atms())
    if get_idxs_of_nonaro_rng_atms == num_nonaro_rngs_orig:
        # Same number of nonaromatic rings as original molecule. Saves the
        # good ones.
        return taut
    else:
        Utils.log(
            "\t"
            + taut.smiles(True)
            + ", a tautomer generated "
            + "from "
            + contnr.orig_smi
            + " ("
            + taut.name
            + "), broke an aromatic ring, so I'm discarding it."
        )


def parallel_check_chiral_centers(taut, contnr):
    """A parallelizable helper function that checks that tautomers do not break
       any chiral centers in the original molecule.

    :param taut: The tautomer to evaluate.
    :type taut: MyMol.MyMol
    :param contnr: The original molecule container.
    :type contnr: MolContainer.MolContainer
    :return: Either the tautomer or a None object.
    :rtype: MyMol.MyMol | None
    """

    # How many chiral centers in the original smiles?
    num_specif_chiral_cntrs_orig = (
        contnr.num_specif_chiral_cntrs + contnr.num_unspecif_chiral_cntrs
    )

    # Make a new list containing only the ones that don't break chiral centers
    # (or that add new chiral centers).
    m_num_specif_chiral_cntrs = len(taut.chiral_cntrs_only_asignd()) + len(
        taut.chiral_cntrs_w_unasignd()
    )
    if m_num_specif_chiral_cntrs == num_specif_chiral_cntrs_orig:
        # Same number of chiral centers as original molecule. Save this good
        # one.
        return taut
    else:
        Utils.log(
            "\t"
            + contnr.orig_smi
            + " ==> "
            + taut.smiles(True)
            + " (tautomer transformation on "
            + taut.name
            + ") "
            + "changed the molecules total number of specified "
            + "chiral centers from "
            + str(num_specif_chiral_cntrs_orig)
            + " to "
            + str(m_num_specif_chiral_cntrs)
            + ", so I'm deleting it."
        )


def parallel_check_carbon_hydrogens(taut, contnr):
    """A parallelizable helper function that checks that tautomers do not
       change the hydrogens on inappropriate carbons.

    :param taut: The tautomer to evaluate.
    :type taut: MyMol.MyMol
    :param contnr: The original molecule container.
    :type contnr: MolContainer.MolContainer
    :return: Either the tautomer or a None object.
    :rtype: MyMol.MyMol | None
    """

    # What's the carbon-hydrogen fingerprint of the original smiles?
    orig_carbon_hydrogen_count = contnr.carbon_hydrogen_count

    # How about this tautomer?
    this_carbon_hydrogen_count = taut.count_hyd_bnd_to_carb()

    # Only keep if they are the same.
    if orig_carbon_hydrogen_count == this_carbon_hydrogen_count:
        return taut
    else:
        Utils.log(
            "\t"
            + contnr.orig_smi
            + " ==> "
            + taut.smiles(True)
            + " (tautomer transformation on "
            + taut.name
            + ") "
            + "changed the number of hydrogen atoms bound to a "
            + "carbon, so I'm deleting it."
        )
