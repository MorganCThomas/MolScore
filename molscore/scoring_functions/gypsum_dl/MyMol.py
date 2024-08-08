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
This module contains classes and functions for processing individual molecules
(variants). All variants of the same input molecule are grouped together in
the same MolContainer.MolContainer object. Each MyMol.MyMol is also associated
with conformers described here (3D coordinate sets).

So just to clarify: MolContainer.MolContainer > MyMol.MyMol >
MyMol.MyConformer
"""

import __future__

import sys
import copy
import operator

import molscore.scoring_functions.gypsum_dl.Utils as Utils
import molscore.scoring_functions.gypsum_dl.MolObjectHandling as MOH

# Disable the unnecessary RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    import rdkit
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondStereo
except:
    Utils.exception("You need to install rdkit and its dependencies.")

try:
    from molscore.scoring_functions.gypsum_dl.molvs import standardize_smiles as ssmiles
except:
    Utils.exception("You need to install molvs and its dependencies.")


class MyMol:
    """
    A class that wraps around a rdkit.Mol object. Includes additional data and
    functions.
    """

    def __init__(self, starter, name=""):
        """Initialize the MyMol object.

        :param starter: The object (smiles or rdkit.Mol) on which to build this
           class.
        :type starter: str or rdkit.Mol
        :param name: An optional string, the name of this molecule. Defaults to "".
        :param name: str, optional
        """

        if isinstance(starter, str):
            # It's a SMILES string.
            self.rdkit_mol = ""
            self.can_smi = ""
            smiles = starter
        else:
            # So it's an rdkit mol object.
            self.rdkit_mol = (
                starter  # No need to regenerate this, since already provided.
            )

            # Get the smiles too from the rdkit mol object.
            try:
                smiles = Chem.MolToSmiles(
                    self.rdkit_mol, isomericSmiles=True, canonical=True
                )

                # In this case you know it's cannonical.
                self.can_smi = smiles
            except:
                # Sometimes this conversion just can't happen. Happened once
                # with this beast, for example:
                # CC(=O)NC1=CC(=C=[N+]([O-])O)C=C1O
                self.can_smi = False
                id_to_print = name if name != "" else str(starter)
                Utils.log(
                    "\tERROR: Could not generate one of the structures "
                    + "for ("
                    + id_to_print
                    + ")."
                )

        self.can_smi_noh = ""
        self.orig_smi = smiles

        # Default assumption is that they are the same.
        self.orig_smi_deslt = smiles
        self.name = name
        self.conformers = []
        self.nonaro_ring_atom_idx = ""
        self.chiral_cntrs_only_assigned = ""
        self.chiral_cntrs_include_unasignd = ""
        self.bizarre_substruct = ""
        self.enrgy = {}  # different energies for different conformers.
        self.minimized_enrgy = {}
        self.contnr_idx = ""
        self.frgs = ""
        self.stdrd_smiles = ""
        self.mol_props = {}
        self.idxs_low_energy_confs_no_opt = {}
        self.idxs_of_confs_to_min = set([])
        self.genealogy = []  # Keep track of how the molecule came to be.

        # Makes the molecule if a smiles was provided. Sanitizes the molecule
        # regardless.
        self.make_mol_frm_smiles_sanitze()

    def standardize_smiles(self):
        """Standardize the smiles string if you can."""

        if self.stdrd_smiles != "":
            return self.stdrd_smiles

        try:
            self.stdrd_smiles = ssmiles(self.smiles())
        except:
            Utils.log("\tCould not standardize " + self.smiles(True) + ". Skipping.")
            self.stdrd_smiles = self.smiles()

        return self.stdrd_smiles

    def __hash__(self):
        """Allows you to compare MyMol.MyMol objects.

        :return: The hashed canonical smiles.
        :rtype: str
        """

        can_smi = self.smiles()

        # So it hashes based on the cannonical smiles.
        return hash(can_smi)

    def __eq__(self, other):
        """Allows you to compare MyMol.MyMol objects.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: Whether the other molecule is the same as this one.
        :rtype: bool
        """

        if other is None:
            return False
        else:
            return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        """Allows you to compare MyMol.MyMol objects.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: Whether the other molecule is different from this one.
        :rtype: bool
        """

        return not self.__eq__(other)

    def __lt__(self, other):
        """Is this MyMol less than another one? Gypsum-DL often sorts
        molecules by sorting tuples of the form (energy, MyMol). On rare
        occasions, the energies are identical, and the sorting algorithm
        attempts to compare MyMol directly.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: True or False, if less than or not.
        :rtype: boolean
        """

        return self.__hash__() < other.__hash__()

    def __le__(self, other):
        """Is this MyMol less than or equal to another one? Gypsum-DL often
        sorts molecules by sorting tuples of the form (energy, MyMol). On rare
        occasions, the energies are identical, and the sorting algorithm
        attempts to compare MyMol directly.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: True or False, if less than or equal to, or not.
        :rtype: boolean
        """

        return self.__hash__() <= other.__hash__()

    def __gt__(self, other):
        """Is this MyMol greater than another one? Gypsum-DL often sorts
        molecules by sorting tuples of the form (energy, MyMol). On rare
        occasions, the energies are identical, and the sorting algorithm
        attempts to compare MyMol directly.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: True or False, if greater than or not.
        :rtype: boolean
        """

        return self.__hash__() > other.__hash__()

    def __ge__(self, other):
        """Is this MyMol greater than or equal to another one? Gypsum-DL often
        sorts molecules by sorting tuples of the form (energy, MyMol). On rare
        occasions, the energies are identical, and the sorting algorithm
        attempts to compare MyMol directly.

        :param other: The other molecule.
        :type other: MyMol.MyMol
        :return: True or False, if greater than or equal to, or not.
        :rtype: boolean
        """

        return self.__hash__() >= other.__hash__()

    def make_mol_frm_smiles_sanitze(self):
        """Construct a rdkit.mol for this object, in case you only received
        the smiles. Also, sanitize the molecule regardless.

        :return: Returns the rdkit.mol object, though it's also stored in
           self.rdkit_mol.
        :rtype: rdkit.mol object.
        """

        # If given a SMILES string.
        if self.rdkit_mol == "":
            try:
                # sanitize = False makes it respect double-bond stereochemistry
                m = Chem.MolFromSmiles(self.orig_smi_deslt, sanitize=False)
            except:
                m = None
        else:  # If given a RDKit Mol Obj
            m = self.rdkit_mol

        if m is not None:
            # Sanitize and hopefully correct errors in the smiles string such
            # as incorrect nitrogen charges.
            m = MOH.check_sanitization(m)
        self.rdkit_mol = m
        return m

    def make_first_3d_conf_no_min(self):
        """Makes the associated rdkit.mol object 3D by adding the first
           conformer. This also adds hydrogen atoms to the associated rdkit.mol
           object. Note that it does not perform a minimization, so it is not
           too expensive."""

        # Set the first 3D conformer
        if len(self.conformers) > 0:
            # It's already been done.
            return

        # Add hydrogens. This adds explicit hydrogens, while respecting
        # Dimorphite-DL protonation states.
        self.rdkit_mol = MOH.try_reprotanation(self.rdkit_mol)

        # Add a single conformer. RMSD cutoff very small so all conformers
        # will be accepted. And not minimizing (False).
        self.add_conformers(1, 1e60, False)

    def smiles(self, noh=False):
        """Get the desalted, canonical smiles string associated with this
           object. (Not the input smiles!)

        :param noh: Whether or not hydrogen atoms should be included in the
           canonical smiles string., defaults to False
        :param noh: bool, optional
        :return: The canonical smiles string, or None if it cannot be
           determined.
        :rtype: str or None
        """

        # See if it's already been calculated.
        if noh == False:
            # They want the hydrogen atoms.
            if self.can_smi != "":
                # Return previously determined canonical SMILES.
                return self.can_smi
            else:
                # Need to determine canonical SMILES.
                try:
                    can_smi = Chem.MolToSmiles(
                        self.rdkit_mol, isomericSmiles=True, canonical=True
                    )
                except:
                    # Sometimes this conversion just can't happen. Happened
                    # once with this beast, for example:
                    # CC(=O)NC1=CC(=C=[N+]([O-])O)C=C1O
                    Utils.log(
                        "Warning: Couldn't put "
                        + self.orig_smi
                        + " ("
                        + self.name
                        + ") in canonical form. Got this error: "
                        + str(sys.exc_info()[0])
                        + ". This molecule will be "
                        + "discarded."
                    )
                    self.can_smi = None
                    return None

                self.can_smi = can_smi
                return can_smi
        else:
            # They don't want the hydrogen atoms.
            if self.can_smi_noh != "":
                # Return previously determined string.
                return self.can_smi_noh

            # So remove hydrogens. Note that this assumes you will have called
            # this function previously with noh = False
            amol = copy.copy(self.rdkit_mol)
            amol = MOH.try_deprotanation(amol)
            self.can_smi_noh = Chem.MolToSmiles(
                amol, isomericSmiles=True, canonical=True
            )
            return self.can_smi_noh

    def get_idxs_of_nonaro_rng_atms(self):
        """Identifies which rings in a given molecule are nonaromatic, if any.

        :return: A [[int, int, int]]. A list of lists, where each inner list is
           a list of the atom indecies of the members of a non-aromatic ring.
           Also saved to self.nonaro_ring_atom_idx.
        :rtype: list
        """

        if self.nonaro_ring_atom_idx != "":
            # Already determined...
            return self.nonaro_ring_atom_idx

        # There are no rings if the molecule is None.
        if self.rdkit_mol is None:
            return []

        # Get the number of symmetric smallest set of rings
        ssr = Chem.GetSymmSSSR(self.rdkit_mol)

        # Get the rings
        ring_indecies = [list(ssr[i]) for i in range(len(ssr))]

        # Are the atoms in any of those rings nonaromatic?
        nonaro_rngs = []
        for rng_indx_set in ring_indecies:
            for atm_idx in rng_indx_set:
                if self.rdkit_mol.GetAtomWithIdx(atm_idx).GetIsAromatic() == False:
                    # One of the ring atoms is not aromatic! Let's keep it.
                    nonaro_rngs.append(rng_indx_set)
                    break
        self.nonaro_ring_atom_idx = nonaro_rngs
        return nonaro_rngs

    def chiral_cntrs_w_unasignd(self):
        """Get the chiral centers that haven't been assigned.

        :return: The chiral centers. Also saved to
           self.chiral_cntrs_include_unasignd. Looks like [(10, '?')]
        :rtype: list
        """

        # No chiral centers if the molecule is None.
        if self.rdkit_mol is None:
            return []

        if self.chiral_cntrs_include_unasignd != "":
            # Already been determined...
            return self.chiral_cntrs_include_unasignd

        # Get the chiral centers that are not defined.
        ccs = Chem.FindMolChiralCenters(self.rdkit_mol, includeUnassigned=True)
        self.chiral_cntrs_include_unasignd = ccs
        return ccs

    def chiral_cntrs_only_asignd(self):
        """Get the chiral centers that have been assigned.

        :return: The chiral centers. Also saved to self.chiral_cntrs_only_assigned.
        :rtype: list
        """

        if self.chiral_cntrs_only_assigned != "":
            return self.chiral_cntrs_only_assigned

        if self.rdkit_mol is None:
            return []

        ccs = Chem.FindMolChiralCenters(self.rdkit_mol, includeUnassigned=False)
        self.chiral_cntrs_only_assigned = ccs
        return ccs

    def get_double_bonds_without_stereochemistry(self):
        """Get the double bonds that don't have specified stereochemistry.

        :return: The unasignd double bonds (indexes). Looks like this:
           [2, 4, 7]
        :rtype: list
        """

        if self.rdkit_mol is None:
            return []

        unasignd = []
        for b in self.rdkit_mol.GetBonds():
            if b.GetBondTypeAsDouble() == 2 and b.GetStereo() is BondStereo.STEREONONE:

                unasignd.append(b.GetIdx())
        return unasignd

    def remove_bizarre_substruc(self):
        """Removes molecules with improbable substuctures, likely generated
           from the tautomerization process. Used to find artifacts.

        :return: Boolean, whether or not there are impossible substructures.
           Also saves to self.bizarre_substruct.
        :rtype: bool
        """

        if self.bizarre_substruct != "":
            # Already been determined.
            return self.bizarre_substruct

        if self.rdkit_mol is None:
            # It is bizarre to have a molecule with no atoms in it.
            return True

        # These are substrutures that can't be easily corrected using
        # fix_common_errors() below.
        # , "[C+]", "[C-]", "[c+]", "[c-]", "[n-]", "[N-]"] # ,
        # "[*@@H]1(~[*][*]~2)~[*]~[*]~[*@@H]2~[*]~[*]~1",
        # "[*@@H]1~2~*~*~[*@@H](~*~*2)~*1",
        # "[*@@H]1~2~*~*~*~[*@@H](~*~*2)~*1",
        # "[*@@H]1~2~*~*~*~*~[*@@H](~*~*2)~*1",
        # "[*@@H]1~2~*~[*@@H](~*~*2)~*1", "[*@@H]~1~2~*~*~*~[*@H]1O2",
        # "[*@@H]~1~2~*~*~*~*~[*@H]1O2"]

        # Note that C(O)=N, C and N mean they are aliphatic. Does not match
        # c(O)n, when aromatic. So this form is acceptable if in aromatic
        # structure.
        prohibited_substructures = ["O(=*)-*"]  # , "C(O)=N"]
        prohibited_substructures.append(
            "C(=[CH2])[OH]"
        )  # Enol forms with terminal alkenes are unlikely.
        prohibited_substructures.append(
            "C(=[CH2])[O-]"
        )  # Enol forms with terminal alkenes are unlikely.
        prohibited_substructures.append(
            "C=C([OH])[OH]"
        )  # A geminal vinyl diol is not a tautomer of a carboxylate group.
        prohibited_substructures.append(
            "C=C([O-])[OH]"
        )  # A geminal vinyl diol is not a tautomer of a carboxylate group.
        prohibited_substructures.append(
            "C=C([O-])[O-]"
        )  # A geminal vinyl diol is not a tautomer of a carboxylate group.
        prohibited_substructures.append("[C-]")  # No carbanions.
        prohibited_substructures.append("[c-]")  # No carbanions.

        for s in prohibited_substructures:
            # First just match strings... could be faster, but not 100%
            # accurate.
            if s in self.orig_smi:
                Utils.log("\tDetected unusual substructure: " + s)
                self.bizarre_substruct = True
                return True

            if s in self.orig_smi_deslt:
                Utils.log("\tDetected unusual substructure: " + s)
                self.bizarre_substruct = True
                return True

            if s in self.can_smi:
                Utils.log("\tDetected unusual substructure: " + s)
                self.bizarre_substruct = True
                return True

        # Now do actual substructure matching
        for s in prohibited_substructures:
            pattrn = Chem.MolFromSmarts(s)
            if self.rdkit_mol.HasSubstructMatch(pattrn):
                # Utils.log("\tRemoving a molecule because it has an odd
                # substructure: " + s)
                Utils.log("\tDetected unusual substructure: " + s)
                self.bizarre_substruct = True
                return True

        # Now certin patterns that are more complex.
        # TODO in the future?

        self.bizarre_substruct = False
        return False

    def get_frags_of_orig_smi(self):
        """Divide the current molecule into fragments.

        :return: A list of the fragments, as rdkit.Mol objects.
        :rtype: list
        """

        if self.frgs != "":
            # Already been determined...
            return self.frgs

        if not "." in self.orig_smi:
            # There are no fragments. Just return this object.
            self.frgs = [self]
            return [self]

        # Get the fragments.
        frags = Chem.GetMolFrags(self.rdkit_mol, asMols=True)
        self.frgs = frags
        return frags

    def inherit_contnr_props(self, other):
        """Copies a few key properties from a different MyMol.MyMol object to
           this one.

        :param other: The other MyMol.MyMol object to copy these properties to.
        :type other: MyMol.MyMol
        """

        # other can be a contnr or MyMol.MyMol object. These are properties
        # that should be the same for every MyMol.MyMol object in this
        # MolContainer.
        self.contnr_idx = other.contnr_idx
        self.orig_smi = other.orig_smi
        self.orig_smi_deslt = other.orig_smi_deslt  # initial assumption
        self.name = other.name

    def set_rdkit_mol_prop(self, key, val):
        """Set a molecular property.

        :param key: The name of the molecular property.
        :type key: str
        :param val: The value of that property.
        :type val: str
        """

        val = str(val)
        self.rdkit_mol.SetProp(key, val)
        self.rdkit_mol.SetProp(key, val)

        try:
            self.rdkit_mol.SetProp(key, val)
        except:
            pass

    def set_all_rdkit_mol_props(self):
        """Set all the stored molecular properties. Copies ones from the
           MyMol.MyMol object to the MyMol.rdkit_mol object."""

        self.set_rdkit_mol_prop("SMILES", self.smiles(True))
        # self.set_rdkit_mol_prop("SOURCE_SMILES", self.orig_smi)
        for prop in list(self.mol_props.keys()):
            self.set_rdkit_mol_prop(prop, self.mol_props[prop])
        genealogy = "\n".join(self.genealogy)
        self.set_rdkit_mol_prop("Genealogy", genealogy)
        self.set_rdkit_mol_prop("_Name", self.name)

    def add_conformers(self, num, rmsd_cutoff=0.1, minimize=True):
        """Add conformers to this molecule.

        :param num: The total number of conformers to generate, including ones
           that have been generated previously.
        :type num: int
        :param rmsd_cutoff: Don't keep conformers that come within this rms
           distance of other conformers. Defaults to 0.1
        :param rmsd_cutoff: float, optional
        :param minimize: Whether or not to minimize the geometry of all these
           conformers. Defaults to True.
        :param minimize: bool, optional
        """

        # First, do you need to add new conformers? Some might have already
        # been added. Just add enough to meet the requested amount.
        num_new_confs = max(0, num - len(self.conformers))
        for i in range(num_new_confs):
            if len(self.conformers) == 0:
                # For the first one, don't start from random coordinates.
                new_conf = MyConformer(self)
            else:
                # For all subsequent ones, do start from random coordinates.
                new_conf = MyConformer(self, None, False, True)

            if new_conf.mol is not False:
                self.conformers.append(new_conf)

        # Are the current ones minimized if necessary?
        if minimize == True:
            for conf in self.conformers:
                conf.minimize()  # Won't reminimize if it's already been done.

        # Automatically sort by the energy.
        self.conformers.sort(key=operator.attrgetter("energy"))

        # Remove ones that are very structurally similar.
        #self.eliminate_structurally_similar_conformers(rmsd_cutoff)

    def eliminate_structurally_similar_conformers(self, rmsd_cutoff=0.1):
        """Eliminates conformers that are very geometrically similar.

        :param rmsd_cutoff: The RMSD cutoff to use. Defaults to 0.1
        :param rmsd_cutoff: float, optional
        """

        # Eliminate redundant ones.
        for i1 in range(0, len(self.conformers) - 1):
            if self.conformers[i1] is not None:
                for i2 in range(i1 + 1, len(self.conformers)):
                    if self.conformers[i2] is not None:
                        # Align them.
                        self.conformers[i2] = self.conformers[i1].align_to_me(
                            self.conformers[i2]
                        )

                        # Calculate the RMSD.
                        rmsd = self.conformers[i1].rmsd_to_me(self.conformers[i2])

                        # Replace the second one with None if it's too similar
                        # to the first.
                        if rmsd <= rmsd_cutoff:
                            self.conformers[i2] = None

        # Remove all the None entries.
        while None in self.conformers:
            self.conformers.remove(None)

        # Those that remains are only the distinct conformers.

    def count_hyd_bnd_to_carb(self):
        """Count the number of Hydrogens bound to carbons."""

        if self.rdkit_mol is None:
            # Doesn't have any atoms at all.
            return 0

        total_hydrogens_counted = 0
        for atom in self.rdkit_mol.GetAtoms():
            if atom.GetSymbol() == "C":
                total_hydrogens_counted = total_hydrogens_counted + atom.GetTotalNumHs(
                    includeNeighbors=True
                )

        return total_hydrogens_counted

    def load_conformers_into_rdkit_mol(self):
        """Load the conformers stored as MyConformers objects (in
           self.conformers) into the rdkit Mol object."""

        self.rdkit_mol.RemoveAllConformers()
        for conformer in self.conformers:
            self.rdkit_mol.AddConformer(conformer.conformer())


class MyConformer:
    """A wrapper around a rdkit Conformer object. Allows me to associate extra
    values with conformers. These are 3D coordinate sets for a given
    MyMol.MyMol object (different molecule conformations).
    """

    def __init__(
        self, mol, conformer=None, second_embed=False, use_random_coordinates=False
    ):
        """Create a MyConformer objects.

        :param mol: The MyMol.MyMol associated with this conformer.
        :type mol: MyMol.MyMol
        :param conformer: An optional variable specifying the conformer to use.
           If not specified, it will create a new conformer. Defaults to None.
        :type conformer: rdkit.Conformer, optional
        :param second_embed: Whether to try to generate 3D coordinates using an
            older algorithm if the better (default) algorithm fails. This can add
            run time, but sometimes converts certain molecules that would
            otherwise fail. Defaults to False.
        :type second_embed: bool, optional
        :param use_random_coordinates: The first conformer should not start
           from random coordinates, but rather the eigenvalues-based
           coordinates rdkit defaults to. But Gypsum-DL generates subsequent
           conformers to try to consider alternate geometries. So they should
           start from random coordinates. Defaults to False.
        :type use_random_coordinates: bool, optional
        """

        # Save some values to the object.
        self.mol = copy.deepcopy(mol.rdkit_mol)
        self.smiles = mol.smiles()

        # Remove any previous conformers.
        self.mol.RemoveAllConformers()

        if conformer is None:
            # The user is providing no conformer. So we must generate it.

            # Note that I have confirmed that the below respects chirality.
            # params is a list of ETKDGv2 parameters generated by this command
            # Description of these parameters can be found at
            # help(AllChem.EmbedMolecule)

            try:
                # Try to use ETKDGv2, but it is only present in the python 3.6
                # version of RDKit.
                params = AllChem.ETKDGv2()
            except:
                # Use the original version of ETKDG if python 2.7 RDKit. This
                # may be resolved in next RDKit update so we encased this in a
                # try statement.
                params = AllChem.ETKDG()

            # The default, but just a sanity check.
            params.enforcechiral = True

            # Set a max number of times it will try to calculate the 3D
            # coordinates. Will save a little time.
            params.maxIterations = 0  # This should be the default but lets
            # set it anyway

            # Also set whether to start from random coordinates.
            params.useRandomCoords = use_random_coordinates

            # AllChem.EmbedMolecule uses geometry to create inital molecule
            # coordinates. This sometimes takes a very long time.
            AllChem.EmbedMolecule(self.mol, params)

            # On rare occasions, the new conformer generating algorithm fails
            # because params.useRandomCoords = False. So if it fails, try
            # again with True.
            if self.mol.GetNumConformers() == 0 and use_random_coordinates == False:
                params.useRandomCoords = True
                AllChem.EmbedMolecule(self.mol, params)

            # On very rare occasions, the new conformer generating algorithm
            # fails. For example, COC(=O)c1cc(C)nc2c(C)cc3[nH]c4ccccc4c3c12 .
            # In this case, the old one still works. So if no coordinates are
            # assigned, try that one. Parameters must have second_embed set to
            # True for this to happen.
            if second_embed == True and self.mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(self.mol, useRandomCoords=use_random_coordinates)

            # On rare occasions, both methods fail. For example,
            # O=c1cccc2[C@H]3C[NH2+]C[C@@H](C3)Cn21 Another example:
            # COc1cccc2c1[C@H](CO)[N@H+]1[C@@H](C#N)[C@@H]3C[C@@H](C(=O)[O-])[C@H]([C@H]1C2)[N@H+]3C
            if self.mol.GetNumConformers() == 0:
                self.mol = False
        else:
            # The user has provided a conformer. Just add it.
            conformer.SetId(0)
            self.mol.AddConformer(conformer, assignId=True)

        # Calculate some energies, other housekeeping.
        if self.mol is not False:
            try:
                ff = AllChem.UFFGetMoleculeForceField(self.mol)
                self.energy = ff.CalcEnergy()
            except:
                Utils.log(
                    "Warning: Could not calculate energy for molecule "
                    + Chem.MolToSmiles(self.mol)
                )
                # Example of smiles that cause problem here without try...catch:
                # NC1=NC2=C(N[C@@H]3[C@H](N2)O[C@@H](COP(O)(O)=O)C2=C3S[Mo](S)(=O)(=O)S2)C(=O)N1
                self.energy = 9999
            self.minimized = False
            self.ids_hvy_atms = [
                a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1
            ]

    def conformer(self, conf=None):
        """Get or set the conformer. An optional variable can specify the
           conformer to set. If not specified, this function acts as a get for
           the conformer.

        :param conf: The conformer to set, defaults to None
        :param conf: rdkit.Conformer, optional
        :return: An rdkit.Conformer object, if conf is not specified.
        :rtype: rdkit.Conformer
        """

        if conf is None:
            return self.mol.GetConformers()[0]
        else:
            self.mol.RemoveAllConformers()
            self.mol.AddConformer(conf)

    def minimize(self):
        """Minimize (optimize) the geometry of the current conformer if it
           hasn't already been optimized."""

        if self.minimized == True:
            # Already minimized. Don't do it again.
            return

        # Perform the minimization, and save the energy.
        try:
            ff = AllChem.UFFGetMoleculeForceField(self.mol)
            ff.Minimize()
            self.energy = ff.CalcEnergy()
        except:
            Utils.log(
                "Warning: Could not calculate energy for molecule "
                + Chem.MolToSmiles(self.mol)
            )
            self.energy = 9999
        self.minimized = True

    def align_to_me(self, other_conf):
        """Align another conformer to this one.

        :param other_conf: The other conformer to align.
        :type other_conf: MyConformer
        :return: The aligned MyConformer object.
        :rtype: MyConformer
        """

        # Add the conformer of the other MyConformer object.
        self.mol.AddConformer(other_conf.conformer(), assignId=True)

        # Align them.
        AllChem.AlignMolConformers(self.mol, atomIds=self.ids_hvy_atms)

        # Reset the conformer of the other MyConformer object.
        last_conf = self.mol.GetConformers()[-1]
        other_conf.conformer(last_conf)

        # Remove the added conformer.
        self.mol.RemoveConformer(last_conf.GetId())

        # Return that other object.
        return other_conf

    def MolToMolBlock(self):
        """Prints out the first 500 letters of the molblock version of this
        conformer. Good for debugging."""

        mol_copy = copy.deepcopy(self.mol_copy)  # Use it as a template.
        mol_copy.RemoveAllConformers()
        mol_copy.AddConformer(self.conformer)
        Utils.log(Chem.MolToMolBlock(mol_copy)[:500])

    def rmsd_to_me(self, other_conf):
        """Calculate the rms distance between this conformer and another one.

        :param other_conf: The other conformer to align.
        :type other_conf: MyConformer
        :return: The rmsd, a float.
        :rtype: float
        """

        # Make a new molecule.
        amol = Chem.MolFromSmiles(self.smiles, sanitize=False)
        amol = MOH.check_sanitization(amol)
        amol = MOH.try_reprotanation(amol)

        # Add the conformer of the other MyConformer object.
        amol.AddConformer(self.conformer(), assignId=True)
        amol.AddConformer(other_conf.conformer(), assignId=True)

        # Get the two confs.
        first_conf = amol.GetConformers()[0]
        last_conf = amol.GetConformers()[-1]

        # Return the RMSD.
        amol = MOH.try_deprotanation(amol)
        rmsd = AllChem.GetConformerRMS(
            amol, first_conf.GetId(), last_conf.GetId(), prealigned=True
        )

        return rmsd
