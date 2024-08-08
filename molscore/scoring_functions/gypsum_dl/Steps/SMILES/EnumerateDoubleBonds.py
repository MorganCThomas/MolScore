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
Module for enumerating unspecified double bonds (cis vs. trans).
"""

import __future__

import itertools
import copy
import random

import molscore.scoring_functions.gypsum_dl.Parallelizer as Parallelizer
import molscore.scoring_functions.gypsum_dl.Utils as Utils
import molscore.scoring_functions.gypsum_dl.ChemUtils as ChemUtils
import molscore.scoring_functions.gypsum_dl.MyMol as MyMol
import math

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


def enumerate_double_bonds(
    contnrs,
    max_variants_per_compound,
    thoroughness,
    num_procs,
    job_manager,
    parallelizer_obj,
):
    """Enumerates all possible cis-trans isomers. If the stereochemistry of a
       double bond is specified, it is not varied. All unspecified double bonds
       are varied.

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
    :param job_manager: The multithred mode to use.
    :type job_manager: string
    :param parallelizer_obj: The Parallelizer object.
    :type parallelizer_obj: Parallelizer.Parallelizer
    """

    # No need to continue if none are requested.
    if max_variants_per_compound == 0:
        return

    Utils.log("Enumerating all possible cis-trans isomers for all molecules...")

    # Group the molecule containers so they can be passed to the parallelizer.
    params = []
    for contnr in contnrs:
        for mol in contnr.mols:
            params.append(tuple([mol, max_variants_per_compound, thoroughness]))
    params = tuple(params)

    # Ruin it through the parallelizer.
    tmp = []
    if parallelizer_obj != None:
        tmp = parallelizer_obj.run(
            params, parallel_get_double_bonded, num_procs, job_manager
        )
    else:
        for i in params:
            tmp.append(parallel_get_double_bonded(i[0], i[1], i[2]))

    # Remove Nones (failed molecules)
    clean = Parallelizer.strip_none(tmp)

    # Flatten the data into a single list.
    flat = Parallelizer.flatten_list(clean)

    # Get the indexes of the ones that failed to generate.
    contnr_idxs_of_failed = Utils.fnd_contnrs_not_represntd(contnrs, flat)

    # Go through the missing ones and throw a message.
    for miss_indx in contnr_idxs_of_failed:
        Utils.log(
            "\tCould not generate valid double-bond variant for "
            + contnrs[miss_indx].orig_smi
            + " ("
            + contnrs[miss_indx].name
            + "), so using existing "
            + "(unprocessed) structures."
        )
        for mol in contnrs[miss_indx].mols:
            mol.genealogy.append("(WARNING: Unable to generate double-bond variant)")
            clean.append(mol)

    flat = ChemUtils.uniq_mols_in_list(flat)

    # Keep only the top few compound variants in each container, to prevent a
    # combinatorial explosion.
    ChemUtils.bst_for_each_contnr_no_opt(
        contnrs, flat, max_variants_per_compound, thoroughness
    )


def parallel_get_double_bonded(mol, max_variants_per_compound, thoroughness):
    """A parallelizable function for enumerating double bonds.

    :param mol: The molecule with a potentially unspecified double bond.
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
    :return: [description]
    :rtype: [type]
    """

    # For this to work, you need to have explicit hydrogens in place.
    mol.rdkit_mol = Chem.AddHs(mol.rdkit_mol)

    # Get all double bonds that don't have defined stereochemistry. Note that
    # these are the bond indexes, not the atom indexes.
    unasignd_dbl_bnd_idxs = mol.get_double_bonds_without_stereochemistry()

    if len(unasignd_dbl_bnd_idxs) == 0:
        # There are no unassigned double bonds, so move on.
        return [mol]

    # Throw out any bond that is in a small ring.
    unasignd_dbl_bnd_idxs = [
        i
        for i in unasignd_dbl_bnd_idxs
        if not mol.rdkit_mol.GetBondWithIdx(i).IsInRingSize(3)
    ]
    unasignd_dbl_bnd_idxs = [
        i
        for i in unasignd_dbl_bnd_idxs
        if not mol.rdkit_mol.GetBondWithIdx(i).IsInRingSize(4)
    ]
    unasignd_dbl_bnd_idxs = [
        i
        for i in unasignd_dbl_bnd_idxs
        if not mol.rdkit_mol.GetBondWithIdx(i).IsInRingSize(5)
    ]
    unasignd_dbl_bnd_idxs = [
        i
        for i in unasignd_dbl_bnd_idxs
        if not mol.rdkit_mol.GetBondWithIdx(i).IsInRingSize(6)
    ]
    unasignd_dbl_bnd_idxs = [
        i
        for i in unasignd_dbl_bnd_idxs
        if not mol.rdkit_mol.GetBondWithIdx(i).IsInRingSize(7)
    ]

    # Previously, I fully enumerated all double bonds. When there are many
    # such bonds, that leads to a combinatorial explosion that causes problems
    # in terms of speed and memory. Now, enumerate only enough bonds to make
    # sure you generate at least thoroughness * max_variants_per_compound.
    unasignd_dbl_bnd_idxs_orig_count = len(unasignd_dbl_bnd_idxs)
    num_bonds_to_keep = int(math.ceil(math.log(thoroughness * max_variants_per_compound, 2)))
    random.shuffle(unasignd_dbl_bnd_idxs)
    unasignd_dbl_bnd_idxs = sorted(unasignd_dbl_bnd_idxs[:num_bonds_to_keep])

    # Get a list of all the single bonds that come off each double-bond atom.
    all_sngl_bnd_idxs = set([])
    dbl_bnd_count = 0
    for dbl_bnd_idx in unasignd_dbl_bnd_idxs:
        bond = mol.rdkit_mol.GetBondWithIdx(dbl_bnd_idx)

        atom1 = bond.GetBeginAtom()
        atom1_bonds = atom1.GetBonds()
        if len(atom1_bonds) == 1:
            # The only bond is the one you already know about. So don't save.
            continue

        atom2 = bond.GetEndAtom()
        atom2_bonds = atom2.GetBonds()
        if len(atom2_bonds) == 1:
            # The only bond is the one you already know about. So don't save.
            continue

        dbl_bnd_count = dbl_bnd_count + 1

        # Suffice it to say, RDKit does not deal with cis-trans isomerization
        # in an intuitive way...
        idxs_of_other_bnds_frm_atm1 = [b.GetIdx() for b in atom1.GetBonds()]
        idxs_of_other_bnds_frm_atm1.remove(dbl_bnd_idx)

        idxs_of_other_bnds_frm_atm2 = [b.GetIdx() for b in atom2.GetBonds()]
        idxs_of_other_bnds_frm_atm2.remove(dbl_bnd_idx)

        all_sngl_bnd_idxs |= set(idxs_of_other_bnds_frm_atm1)
        all_sngl_bnd_idxs |= set(idxs_of_other_bnds_frm_atm2)

    # Now come up with all possible up/down combinations for those bonds.
    all_sngl_bnd_idxs = list(all_sngl_bnd_idxs)
    all_atom_config_options = list(
        itertools.product([True, False], repeat=len(all_sngl_bnd_idxs))
    )

    # Let the user know.
    if dbl_bnd_count > 0:
        Utils.log(
            "\t"
            + mol.smiles(True)
            + " has "
            # + str(dbl_bnd_count)
            + str(
                # Not exactly right, I think, because should be dbl_bnd_count, but ok.
                unasignd_dbl_bnd_idxs_orig_count
            )
            + " double bond(s) with unspecified stereochemistry."
        )

    # Go through and consider each of the retained combinations.
    smiles_to_consider = set([])
    for atom_config_options in all_atom_config_options:
        # Make a copy of the original RDKit molecule.
        a_rd_mol = copy.copy(mol.rdkit_mol)
        # a_rd_mol = Chem.MolFromSmiles(mol.smiles())

        for bond_idx, direc in zip(all_sngl_bnd_idxs, atom_config_options):
            # Always done with reference to the atom in the double bond.
            if direc:
                a_rd_mol.GetBondWithIdx(bond_idx).SetBondDir(Chem.BondDir.ENDUPRIGHT)
            else:
                a_rd_mol.GetBondWithIdx(bond_idx).SetBondDir(Chem.BondDir.ENDDOWNRIGHT)

        # Assign the StereoChemistry. Required to actually set it.
        a_rd_mol.ClearComputedProps()
        Chem.AssignStereochemistry(a_rd_mol, force=True)

        # Add to list of ones to consider
        try:
            smiles_to_consider.add(
                Chem.MolToSmiles(a_rd_mol, isomericSmiles=True, canonical=True)
            )
        except:
            # Some molecules still give troubles. Unfortunate, but these are
            # rare cases. Let's just skip these. Example:
            # CN1C2=C(C=CC=C2)C(C)(C)[C]1=[C]=[CH]C3=CC(=C(O)C(=C3)I)I
            continue

    # Remove ones that don't have "/" or "\". These are not real enumerated ones.
    smiles_to_consider = [s for s in smiles_to_consider if "/" in s or "\\" in s]

    # Get the maximum number of / + \ in any string.
    cnts = [s.count("/") + s.count("\\") for s in smiles_to_consider]

    if len(cnts) == 0:
        # There are no appropriate double bonds. Move on...
        return [mol]

    max_cnts = max(cnts)

    # Only keep those with that same max count. The others have double bonds
    # that remain unspecified.
    smiles_to_consider = [
        s[0] for s in zip(smiles_to_consider, cnts) if s[1] == max_cnts
    ]
    results = []
    for smile_to_consider in smiles_to_consider:
        # Make a new MyMol.MyMol object with the specified smiles.
        new_mol = MyMol.MyMol(smile_to_consider)

        if new_mol.can_smi != False and new_mol.can_smi != None:
            # Sometimes you get an error if there's a bad structure otherwise.

            # Add the new molecule to the list of results, if it does not have
            # a bizarre substructure.
            if not new_mol.remove_bizarre_substruc():
                new_mol.contnr_idx = mol.contnr_idx
                new_mol.name = mol.name
                new_mol.genealogy = mol.genealogy[:]
                new_mol.genealogy.append(
                    new_mol.smiles(True) + " (cis-trans isomerization)"
                )
                results.append(new_mol)

    # Return the results.
    return results
