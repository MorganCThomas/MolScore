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
This module describes the MolContainer, which contains different MyMol.MyMol
objects. Each object in this container is derived from the same input molecule
(so they are variants). Note that conformers (3D coordinate sets) live inside
MyMol.MyMol. So, just to clarify:

MolContainer.MolContainer > MyMol.MyMol > MyMol.MyConformers
"""

import __future__

import gypsum_dl.Utils as Utils
import gypsum_dl.ChemUtils as ChemUtils
import gypsum_dl.MyMol as MyMol

try:
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")


class MolContainer:
    """The molecucle container class. It stores all the molecules (tautomers,
    etc.) associated with a single input SMILES entry."""

    def __init__(self, smiles, name, index, properties):
        """The constructor.

        :param smiles: A list of SMILES strings.
        :type smiles: str
        :param name: The name of the molecule.
        :type name: str
        :param index: The index of this MolContainer in the main MolContainer
           list.
        :type index: int
        :param properties: A dictionary of properties from the sdf.
        :type properties: dict
        """

        # Set some variables are set on the container level (not the MyMol
        # level)
        self.contnr_idx = index
        self.contnr_idx_orig = index  # Because if some circumstances (mpi),
        # might be reset. But good to have
        # original for filename output.
        self.orig_smi = smiles
        self.orig_smi_deslt = smiles  # initial assumption
        self.mols = []
        self.name = name
        self.properties = properties
        self.mol_orig_frm_inp_smi = MyMol.MyMol(smiles, name)
        self.mol_orig_frm_inp_smi.contnr_idx = self.contnr_idx
        self.frgs = ""  # For caching.

        # Save the original canonical smiles
        self.orig_smi_canonical = self.mol_orig_frm_inp_smi.smiles()

        # Get the number of nonaromatic rings
        self.num_nonaro_rngs = len(
            self.mol_orig_frm_inp_smi.get_idxs_of_nonaro_rng_atms()
        )

        # Get the number of chiral centers, assigned
        self.num_specif_chiral_cntrs = len(
            self.mol_orig_frm_inp_smi.chiral_cntrs_only_asignd()
        )

        # Also get the number of chiral centers, unassigned
        self.num_unspecif_chiral_cntrs = len(
            self.mol_orig_frm_inp_smi.chiral_cntrs_w_unasignd()
        )

        # Get the non-acidic carbon-hydrogen footprint.
        self.carbon_hydrogen_count = self.mol_orig_frm_inp_smi.count_hyd_bnd_to_carb()

    def mol_with_smiles_is_in_contnr(self, smiles):
        """Checks whether or not a given smiles string is already in this
           container.

        :param smiles: The smiles string to check.
        :type smiles: str
        :return: True if it is present, otherwise a new MyMol.MyMol object
           corresponding to that smiles.
        :rtype: bool or MyMol.MyMol
        """

        # Checks all the mols in this container to see if a given smiles is
        # already present. Returns a new MyMol object if it isn't, True
        # otherwise.

        # First, get the set of all cannonical smiles.
        # TODO: Probably shouldn't be generating this on the fly every time
        # you use it!
        can_smi_in_this_container = set([m.smiles() for m in self.mols])

        # Determine whether it is already in the container, and act
        # accordingly.
        amol = MyMol.MyMol(smiles)
        if amol.smiles() in can_smi_in_this_container:
            return True
        else:
            return amol

    def add_smiles(self, smiles):
        """Adds smiles strings to this container. SMILES are always isomeric
           and always unique (canonical).

        :param smiles: A list of SMILES strings. If it's a string, it is
           converted into a list.
        :type smiles: str
        """

        # Convert it into a list if it comes in as a string.
        if isinstance(smiles, str):
            smiles = [smiles]

        # Keep only the mols with smiles that are not already present.
        for s in smiles:
            result = self.mol_with_smiles_is_in_contnr(s)
            if result != True:
                # Much of the contnr info should be passed to each molecule,
                # too, for convenience.
                result.name = self.name
                result.name = self.orig_smi
                result.orig_smi_canonical = self.orig_smi_canonical
                result.orig_smi_deslt = self.orig_smi_deslt
                result.contnr_idx = self.contnr_idx

                self.mols.append(result)

    def add_mol(self, mol):
        """Adds a molecule to this container. Does NOT check for uniqueness.

        :param mol: The MyMol.MyMol object to add.
        :type mol: MyMol.MyMol
        """

        self.mols.append(mol)

    def all_can_noh_smiles(self):
        """Gets a list of all the noh canonical smiles in this container.

        :return: The canonical, noh smiles string.
        :rtype: str
        """

        smiles = []
        for m in self.mols:
            if m.rdkit_mol is not None:
                smiles.append(m.smiles(True))  # True means noh

        return smiles

    def get_frags_of_orig_smi(self):
        """Gets a list of the fragments found in the original smiles string
           passed to this container.

        :return: A list of the fragments, as rdkit.Mol objects. Also saves to
           self.frgs.
        :rtype: list
        """

        if self.frgs != "":
            return self.frgs

        frags = self.mol_orig_frm_inp_smi.get_frags_of_orig_smi()
        self.frgs = frags
        return frags

    def update_orig_smi(self, orig_smi):
        """Updates the orig_smi string. Used by desalter (to replace with
           largest fragment).

        :param orig_smi: The replacement smiles string.
        :type orig_smi: str
        """

        # Update the MolContainer object
        self.orig_smi = orig_smi
        self.orig_smi_deslt = orig_smi
        self.mol_orig_frm_inp_smi = MyMol.MyMol(self.orig_smi, self.name)
        self.frgs = ""
        self.orig_smi_canonical = self.mol_orig_frm_inp_smi.smiles()
        self.num_nonaro_rngs = len(
            self.mol_orig_frm_inp_smi.get_idxs_of_nonaro_rng_atms()
        )
        self.num_specif_chiral_cntrs = len(
            self.mol_orig_frm_inp_smi.chiral_cntrs_only_asignd()
        )
        self.num_unspecif_chiral_cntrs = len(
            self.mol_orig_frm_inp_smi.chiral_cntrs_w_unasignd()
        )

        # None of the mols derived to date, if present, are accurate.
        self.mols = []

    def add_container_properties(self):
        """Adds all properties from the container to the molecules. Used when
           saving final files, to keep a record in the file itself."""

        for mol in self.mols:
            mol.mol_props.update(self.properties)
            mol.set_all_rdkit_mol_props()

    def remove_identical_mols_from_contnr(self):
        """Removes itentical molecules from this container."""

        # For reasons I don't understand, the following doesn't give unique
        # canonical smiles:

        # Chem.MolToSmiles(self.mols[0].rdkit_mol, isomericSmiles=True,
        # canonical=True)

        # # This block for debugging. JDD: Needs attention?
        # all_can_noh_smiles = [m.smiles() for m in self.mols]  # Get all the smiles as stored.

        # wrong_cannonical_smiles = [
        #     Chem.MolToSmiles(
        #         m.rdkit_mol,  # Using the RdKit mol stored in MyMol
        #         isomericSmiles=True,
        #         canonical=True
        #     ) for m in self.mols
        # ]

        # right_cannonical_smiles = [
        #     Chem.MolToSmiles(
        #         Chem.MolFromSmiles(  # Regenerating the RdKit mol from the smiles string stored in MyMol
        #             m.smiles()
        #         ),
        #         isomericSmiles=True,
        #         canonical=True
        #     ) for m in self.mols]

        # if len(set(wrong_cannonical_smiles)) != len(set(right_cannonical_smiles)):
        #     Utils.log("ERROR!")
        #     Utils.log("Stored smiles string in this container:")
        #     Utils.log("\n".join(all_can_noh_smiles))
        #     Utils.log("")
        #     Utils.log("""Supposedly cannonical smiles strings generated from stored
        #         RDKit Mols in this container:""")
        #     Utils.log("\n".join(wrong_cannonical_smiles))
        #     Utils.log("""But if you plop these into chemdraw, you'll see some of them
        #         represent identical structures.""")
        #     Utils.log("")
        #     Utils.log("""Cannonical smiles strings generated from RDKit mols that
        #         were generated from the stored smiles string in this container:""")
        #     Utils.log("\n".join(right_cannonical_smiles))
        #     Utils.log("""Now you see the identical molecules. But why didn't the previous
        #         method catch them?""")
        #     Utils.log("")

        #     Utils.log("""Note that the third method identifies duplicates that the second
        #         method doesn't.""")
        #     Utils.log("")
        #     Utils.log("=" * 20)

        # # You need to make new molecules to get it to work.
        # new_smiles = [m.smiles() for m in self.mols]
        # new_mols = [Chem.MolFromSmiles(smi) for smi in new_smiles]
        # new_can_smiles = [Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=True) for new_mol in new_mols]

        # can_smiles_already_set = set([])
        # for i, new_can_smile in enumerate(new_can_smiles):
        #     if not new_can_smile in can_smiles_already_set:
        #         # Never seen before
        #         can_smiles_already_set.add(new_can_smile)
        #     else:
        #         # Seen before. Delete!
        #         self.mols[i] = None

        # while None in self.mols:
        #     self.mols.remove(None)

        self.mols = ChemUtils.uniq_mols_in_list(self.mols)

    def update_idx(self, new_idx):
        """Updates the index of this container.

        :param new_idx: The new index.
        :type new_idx: int
        """

        if type(new_idx) != int:
            Utils.exception("New idx value must be an int.")
        self.contnr_idx = new_idx
        self.mol_orig_frm_inp_smi.contnr_idx = self.contnr_idx
