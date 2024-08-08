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
Runs the smile preparation process. Generates alternate ionization,
tautomeric, chiral forms, etc.
"""

import __future__

from molscore.scoring_functions.gypsum_dl import Utils
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.DeSaltOrigSmiles import desalt_orig_smi
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.AddHydrogens import add_hydrogens
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.MakeTautomers import make_tauts
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.DurrantLabFilter import (
    durrant_lab_filters,
    durrant_lab_contains_bad_substr,
)
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.EnumerateChiralMols import enumerate_chiral_molecules
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.EnumerateDoubleBonds import enumerate_double_bonds


def prepare_smiles(contnrs, params):
    """Runs the appropriate steps for processing the SMILES strings.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param params: The user parameters.
    :type params: dict
    """

    # Unpack some of the parameter values.
    min_ph = params["min_ph"]
    max_ph = params["max_ph"]
    std_dev = params["pka_precision"]
    max_variants_per_compound = params["max_variants_per_compound"]
    thoroughness = params["thoroughness"]
    num_procs = params["num_processors"]
    job_manager = params["job_manager"]
    let_tautomers_change_chirality = params["let_tautomers_change_chirality"]
    parallelizer_obj = params["Parallelizer"]

    debug = True

    # Desalt the molecules. Note that the program always desalts (can't turn it
    # off).
    # Utils.log("Begin Desaltings")
    desalt_orig_smi(contnrs, num_procs, job_manager, parallelizer_obj)
    # Utils.log("Done with Desalting")

    # Filter the containers to remove ones that have bad substrings (metal,
    # etc.) in the desalted smiles, assuming durrant lab filter turned on. Note
    # that some compounds aren't filtered until later.
    if params["use_durrant_lab_filters"] == True:
        contnrs = [
            c for c in contnrs if not durrant_lab_contains_bad_substr(c.orig_smi_deslt)
        ]

    if debug:
        Utils.print_current_smiles(contnrs)

    # Add hydrogens for user-specified pH, if requested.
    if not params["skip_adding_hydrogen"]:
        # Utils.log("Ionizing Molecules")
        add_hydrogens(
            contnrs,
            min_ph,
            max_ph,
            std_dev,
            max_variants_per_compound,
            thoroughness,
            num_procs,
            job_manager,
            parallelizer_obj,
        )
        # Utils.log("Done with Ionization")
    else:
        Utils.log("Skipping ionization")
        wrap_molecules(contnrs)

    if debug:
        Utils.print_current_smiles(contnrs)

    # Make alternate tautomeric forms, if requested.
    if not params["skip_making_tautomers"]:
        # Utils.log("Tautomerizing Molecules")
        make_tauts(
            contnrs,
            max_variants_per_compound,
            thoroughness,
            num_procs,
            job_manager,
            let_tautomers_change_chirality,
            parallelizer_obj,
        )
        # Utils.log("Done with Tautomerization")
    else:
        Utils.log("Skipping tautomerization")

    if debug:
        Utils.print_current_smiles(contnrs)

    # Apply Durrant-lab filters if requested
    if params["use_durrant_lab_filters"]:
        # Utils.log("Applying Durrant-Lab Filters")
        durrant_lab_filters(contnrs, num_procs, job_manager, parallelizer_obj)
        # Utils.log("Done Applying Durrant-Lab Filters")
    else:
        Utils.log("Not applying Durrant-lab filters")

    if debug:
        Utils.print_current_smiles(contnrs)

    # Make alternate chiral forms, if requested.
    if not params["skip_enumerate_chiral_mol"]:
        # Utils.log("Enumerating Chirality")
        enumerate_chiral_molecules(
            contnrs,
            max_variants_per_compound,
            thoroughness,
            num_procs,
            job_manager,
            parallelizer_obj,
        )
        # Utils.log("Done with Chirality Enumeration")
    else:
        Utils.log("Skipping chirality enumeration")

    if debug:
        Utils.print_current_smiles(contnrs)

    # Make alternate double-bond isomers, if requested.
    if not params["skip_enumerate_double_bonds"]:
        # Utils.log("Enumerating Double Bonds")
        enumerate_double_bonds(
            contnrs,
            max_variants_per_compound,
            thoroughness,
            num_procs,
            job_manager,
            parallelizer_obj,
        )
        # Utils.log("Done with Double Bond Enumeration")
    else:
        Utils.log("Skipping double bond enumeration")

    if debug:
        Utils.print_current_smiles(contnrs)

    return contnrs


def wrap_molecules(contnrs):
    """Each molecule container holds only one SMILES string
    (corresponding to the input structure). Dimorphite-DL can potentially
    produce multiple SMILES strings with different ionization states. There is
    no way to store muliple smiles in a molecule container.

    But those containers are designed to store multiple RDKit molecule
    objects. Gypsum-DL stores Dimorphite-DL output molecules as RdKit molecule
    objects, in the container's mol list.

    But Gypsum-DL gives the user the option of skipping the ionization step.
    In this case, the one SMILES needs to be converted to a RDKit mol object
    for subsequent steps to work. Let's do that here.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    """

    for mol_cont in contnrs:
        if len(mol_cont.mols) == 0:
            smi = mol_cont.orig_smi_canonical
            mol_cont.add_smiles(smi)
