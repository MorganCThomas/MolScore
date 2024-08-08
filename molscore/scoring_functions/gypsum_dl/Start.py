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
Contains the prepare_molecules definition which reads, prepares, and writes
small molecules.
"""

import __future__

import sys
import json
import os
from datetime import datetime
from collections import OrderedDict

import molscore.scoring_functions.gypsum_dl.Utils as Utils
from molscore.scoring_functions.gypsum_dl.Parallelizer import Parallelizer
from molscore.scoring_functions.gypsum_dl.Parallelizer import flatten_list

try:
    from rdkit.Chem import AllChem
    from rdkit import Chem
except:
    Utils.exception("You need to install rdkit and its dependencies.")

try:
    import numpy
except:
    Utils.exception("You need to install numpy and its dependencies.")

try:
    from scipy.cluster.vq import kmeans2
except:
    Utils.exception("You need to install scipy and its dependencies.")

from molscore.scoring_functions.gypsum_dl.MolContainer import MolContainer
from molscore.scoring_functions.gypsum_dl.Steps.SMILES.PrepareSmiles import prepare_smiles
from molscore.scoring_functions.gypsum_dl.Steps.ThreeD.PrepareThreeD import prepare_3d
from molscore.scoring_functions.gypsum_dl.Steps.IO.ProcessOutput import proccess_output
from molscore.scoring_functions.gypsum_dl.Steps.IO.LoadFiles import load_smiles_file
from molscore.scoring_functions.gypsum_dl.Steps.IO.LoadFiles import load_sdf_file

# see http://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules
def prepare_molecules(args):
    """A function for preparing small-molecule models for docking. To work, it
    requires that the python module rdkit be installed on the system.

    :param args: The arguments, from the commandline.
    :type args: dict
    """

    # Keep track of the tim the program starts.
    start_time = datetime.now()

    # A list of command-line parameters that will be ignored if using a json
    # file.
    json_warning_list = [
        "source",
        "output_folder",
        "num_processors",
        "min_ph",
        "max_ph",
        "delta_ph_increment",
        "thoroughness",
        "max_variants_per_compound",
        "pka_precision",
    ]

    # Whether to warn the user that the above parameters, if specified, will
    # be ignored.
    need_to_print_override_warning = False

    if "json" in args:
        # "json" is one of the parameters, so we'll be ignoring the rest.
        try:
            params = json.load(open(args["json"]))
        except:
            Utils.exception("Is your input json file properly formed?")

        params = set_parameters(params)
        if [i for i in json_warning_list if i in list(args.keys())]:
            need_to_print_override_warning = True
    else:
        # We're actually going to use all the command-line parameters. No
        # warning necessary.
        params = set_parameters(args)

    # If running in serial mode, make sure only one processor is used.
    if params["job_manager"] == "serial":
        if params["num_processors"] != 1:
            Utils.log(
                "Because --job_manager was set to serial, this will be run on a single processor."
            )
        params["num_processors"] = 1

    # Handle mpi errors if mpi4py isn't installed
    if params["job_manager"] == "mpi":

        # Before executing Parallelizer with mpi4py (which override python raise Exceptions)
        # We must check that it is being run with the "-m mpi4py" runpy flag
        sys_modules = sys.modules
        if "runpy" not in sys_modules.keys():
            printout = "\nTo run in mpi mode you must run with -m flag. ie) mpirun -n $NTASKS python -m mpi4py run_gypsum_dl.py\n"
            print(printout)
            Utils.exception(printout)

        # Check mpi4py import
        try:
            import mpi4py
        except:
            printout = "\nmpi4py not installed but --job_manager is set to mpi. \n Either install mpi4py or switch job_manager to multiprocessing or serial.\n"
            print(printout)
            Utils.exception(printout)

        # Check mpi4py import version. This must be at version 2.1.0 and higher
        mpi4py_version = mpi4py.__version__
        mpi4py_version = [int(x) for x in mpi4py_version.split(".")]

        if mpi4py_version[0] == 2:
            if mpi4py_version[1] < 1:
                printout = "\nmpi4py version 2.1.0 or higher is required. Use the 'python -m mpi4py' flag to run in mpi mode.\nPlease update mpi4py to a newer version, or switch job_manager to multiprocessing or serial.\n"
                print(printout)
                Utils.exception(printout)
        elif mpi4py_version[0] < 2:
            printout = "\nmpi4py version 2.1.0 or higher is required. Use the 'python -m mpi4py' flag to run in mpi mode.\nPlease update mpi4py to a newer version, or switch job_manager to multiprocessing or serial.\n"
            print(printout)
            Utils.exception(printout)

    # Throw a message if running on windows. Windows doesn't deal with with
    # multiple processors, so use only 1.
    if sys.platform == "win32":
        Utils.log(
            "WARNING: Multiprocessing is not supported on Windows. Tasks will be run in Serial mode."
        )
        params["num_processors"] = 1
        params["job_manager"] = "serial"

    # Launch mpi workers if that's what's specified.
    if params["job_manager"] == "mpi":
        params["Parallelizer"] = Parallelizer(
            params["job_manager"], params["num_processors"]
        )
    else:
        # Lower-level mpi (i.e. making a new Parallelizer within an mpi) has
        # problems with importing the MPI environment and mpi4py. So we will
        # flag it to skip the MPI mode and just go to multiprocess/serial.
        # This is a saftey precaution
        params["Parallelizer"] = Parallelizer(
            params["job_manager"], params["num_processors"], True
        )

    # Let the user know that their command-line parameters will be ignored, if
    # they have specified a json file.
    if need_to_print_override_warning == True:
        Utils.log("WARNING: Using the --json flag overrides all other flags.")

    # If running in mpi mode, separate_output_files must be set to true.
    if params["job_manager"] == "mpi" and params["separate_output_files"] == False:
        Utils.log(
            "WARNING: Running in mpi mode, but separate_output_files is not set to True. Setting separate_output_files to True anyway."
        )
        params["separate_output_files"] = True

    # Outputing HTML files not supported in mpi mode.
    if params["job_manager"] == "mpi" and params["add_html_output"] == True:
        Utils.log(
            "WARNING: Running in mpi mode, but add_html_output is set to True. HTML output is not supported in mpi mode."
        )
        params["add_html_output"] = False

    # Warn the user if he or she is not using the Durrant lab filters.
    if params["use_durrant_lab_filters"] ==- False:
        Utils.log(
            "WARNING: Running Gypsum-DL without the Durrant-lab filters. In looking over many Gypsum-DL-generated " +
            "variants, we have identified a number of substructures that, though technically possible, strike us " +
            "as improbable or otherwise poorly suited for virtual screening. We strongly recommend removing these " +
            "by running Gypsum-DL with the --use_durrant_lab_filters option.",
            trailing_whitespace="\n"
        )

    # Load SMILES data
    if isinstance(params["source"], str):
        Utils.log("Loading molecules from " + os.path.basename(params["source"]) + "...")

        # Smiles must be array of strs.
        src = params["source"]
        if src.lower().endswith(".smi") or src.lower().endswith(".can"):
            # It's an smi file.
            smiles_data = load_smiles_file(src)
        elif params["source"].lower().endswith(".sdf"):
            # It's an sdf file. Convert it to a smiles.
            smiles_data = load_sdf_file(src)
        else:
            smiles_data = [params["source"]]
    else:
        pass  # It's already in the required format.

    # Make the output directory if necessary.
    if os.path.exists(params["output_folder"]) == False:
        os.mkdir(params["output_folder"])
        if os.path.exists(params["output_folder"]) == False:
            Utils.exception("Output folder directory couldn't be found or created.")

    # For Debugging
    # print("")
    # print("###########################")
    # print("num_procs  :  ", params["num_processors"])
    # print("chosen mode  :  ", params["job_manager"])
    # print("Parallel style:  ", params["Parallelizer"].return_mode())
    # print("Number Nodes:  ", params["Parallelizer"].return_node())
    # print("###########################")
    # print("")

    # Make the molecule containers.
    contnrs = []
    idx_counter = 0
    for i in range(0, len(smiles_data)):
        try:
            smiles, name, props = smiles_data[i]
        except:
            msg = 'Unexpected error. Does your "source" parameter specify a '
            msg = msg + "filename that ends in a .can, .smi, or .sdf extension?"
            Utils.exception(msg)

        if detect_unassigned_bonds(smiles) is None:
            Utils.log(
                "WARNING: Throwing out SMILES because of unassigned bonds: " + smiles
            )
            continue

        new_contnr = MolContainer(smiles, name, idx_counter, props)
        if (
            new_contnr.orig_smi_canonical == None
            or type(new_contnr.orig_smi_canonical) != str
        ):
            Utils.log(
                "WARNING: Throwing out SMILES because of it couldn't convert to mol: "
                + smiles
            )
            continue

        contnrs.append(new_contnr)
        idx_counter += 1

    # Remove None types from failed conversion
    contnrs = [x for x in contnrs if x.orig_smi_canonical != None]
    if len(contnrs) != idx_counter:
        Utils.exception("There is a corrupted container")

    # In multiprocessing mode, Gypsum-DL parallelizes each small-molecule
    # preparation step separately. But this scheme is inefficient in MPI mode
    # because it increases the amount of communication required between nodes.
    # So for MPI mode, we will run all the preparation steps for a given
    # molecule container on a single thread.
    if params["Parallelizer"].return_mode() != "mpi":
        # Non-MPI (e.g., multiprocessing)
        execute_gypsum_dl(contnrs, params)
    else:
        # MPI mode. Group the molecule containers so they can be passed to the
        # parallelizer.
        job_input = []
        temp_param = {}
        for key in list(params.keys()):
            if key == "Parallelizer":
                temp_param["Parallelizer"] = None
            else:
                temp_param[key] = params[key]

        for contnr in contnrs:
            contnr.contnr_idx = 0  # Because each container being run in isolation.
            job_input.append(tuple([[contnr], temp_param]))
        job_input = tuple(job_input)

        params["Parallelizer"].run(job_input, execute_gypsum_dl)

    # Calculate the total run time.
    end_time = datetime.now()
    run_time = end_time - start_time
    params["start_time"] = str(start_time)
    params["end_time"] = str(end_time)
    params["run_time"] = str(run_time)

    Utils.log("\nStart time at: " + str(start_time))
    Utils.log("End time at:   " + str(end_time))
    Utils.log("Total time at: " + str(run_time))

    # Kill mpi workers if necessary.
    params["Parallelizer"].end(params["job_manager"])


def execute_gypsum_dl(contnrs, params):
    """A function for doing all of the manipulations to each molecule.

    :param contnrs: A list of all molecules.
    :type contnrs: list
    :param params: A dictionary containing all of the parameters.
    :type params: dict
    """
    # Start creating the models.

    # Prepare the smiles. Desalt, consider alternate ionization, tautometeric,
    # stereoisomeric forms, etc.
    prepare_smiles(contnrs, params)

    # Convert the processed SMILES strings to 3D.
    prepare_3d(contnrs, params)

    # Add in name and unique id to each molecule.
    add_mol_id_props(contnrs)

    # Output the current SMILES.
    Utils.print_current_smiles(contnrs)

    # Write any mols that fail entirely to a file.
    deal_with_failed_molecules(contnrs, params)  ####

    # Process the output.
    proccess_output(contnrs, params)


def detect_unassigned_bonds(smiles):
    """Detects whether a give smiles string has unassigned bonds.

    :param smiles: The smiles string.
    :type smiles: string
    :return: None if it has bad bonds, or the input smiles string otherwise.
    :rtype: None|string
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        # Apparently the bonds are particularly bad, because couldn't even
        # create the molecule.
        return None
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 0:
            return None
    return smiles


def set_parameters(params_unicode):
    """Set the parameters that will control this ConfGenerator object.

    :param params_unicode: The parameters, with keys and values possibly in
       unicode.
    :type params_unicode: dict
    :return: The parameters, properly processed, with defaults used when no
       value specified.
    :rtype: dict
    """

    # Set the default values.
    default = OrderedDict(
        {
            "source": "",
            "output_folder": "./",
            "separate_output_files": False,
            "add_pdb_output": False,
            "add_html_output": False,
            "num_processors": -1,
            "start_time": 0,
            "end_time": 0,
            "run_time": 0,
            "min_ph": 6.4,
            "max_ph": 8.4,
            "pka_precision": 1.0,
            "thoroughness": 3,
            "max_variants_per_compound": 5,
            "second_embed": False,
            "2d_output_only": False,
            "skip_optimize_geometry": False,
            "skip_alternate_ring_conformations": False,
            "skip_adding_hydrogen": False,
            "skip_making_tautomers": False,
            "skip_enumerate_chiral_mol": False,
            "skip_enumerate_double_bonds": False,
            "let_tautomers_change_chirality": False,
            "use_durrant_lab_filters": False,
            "job_manager": "multiprocessing",
            "cache_prerun": False,
            "test": False,
        }
    )

    # Modify params so that they keys are always lower case.
    # Also, rdkit doesn't play nice with unicode, so convert to ascii

    # Because Python2 & Python3 use different string objects, we separate their
    # usecases here.
    params = {}
    if sys.version_info < (3,):
        # For Python3
        for param in params_unicode:
            val = params_unicode[param]
            if isinstance(val, unicode):
                val = str(val).encode("utf8")
            key = param.lower().encode("utf8")
            params[key] = val
    else:
        # For Python2
        for param in params_unicode:
            val = params_unicode[param]
            key = param.lower()
            params[key] = val

    # Overwrites values with the user parameters where they exit.
    merge_parameters(default, params)

    # Checks and prepares the final parameter list.
    final_params = finalize_params(default)

    return final_params


def merge_parameters(default, params):
    """Add default values if missing from parameters.

    :param default: The parameters.
    :type default: dict
    :param params: The default values
    :type params: dict
    :raises KeyError: Unrecognized parameter.
    :raises TypeError: Input parameter has a different type than the default.
    """

    # Generate a dictionary with the same keys, but the types for the values.
    type_dict = make_type_dict(default)

    # Move user-specified values into the parameter.
    for param in params:
        # Throw an error if there's an unrecognized parameter.
        if param not in default:
            Utils.log('Parameter "' + str(param) + '" not recognized!')
            Utils.log("Here are the options:")
            Utils.log(" ".join(sorted(list(default.keys()))))
            Utils.exception("Unrecognized parameter: " + str(param))

        # Throw an error if the input parameter has a different type than
        # the default one.
        if not isinstance(params[param], type_dict[param]):
            # Cast int to float if necessary
            if type(params[param]) is int and type_dict[param] is float:
                params[param] = float(params[param])
            else:
                # Seems to be a type mismatch.
                Utils.exception(
                    'The parameter "'
                    + param
                    + '" must be of '
                    + "type "
                    + str(type_dict[param])
                    + ", but it is of type "
                    + str(type(params[param]))
                    + "."
                )

        # Update the parameter value with the user-defined one.
        default[param] = params[param]


def make_type_dict(dictionary):
    """Creates a types dictionary from an existant dictionary. Keys are
       preserved, but values are the types.

    :param dictionary: A dictionary, with keys are values.
    :type dictionary: dict
    :return: A dictionary with the same keys, but the values are the types.
    :rtype: dict
    """

    type_dict = {}
    allowed_types = [int, float, bool, str]
    # Go through the dictionary keys.
    for key in dictionary:
        # Get the the type of the value.
        val = dictionary[key]
        for allowed in allowed_types:
            if isinstance(val, allowed):
                # Add it to the type_dict.
                type_dict[key] = allowed

        # The value ha san unacceptable type. Throw an error.
        if key not in type_dict:
            Utils.exception(
                "ERROR: There appears to be an error in your parameter "
                + "JSON file. No value can have type "
                + str(type(val))
                + "."
            )

    return type_dict


def finalize_params(params):
    """Checks and updates parameters to their final values.

    :param params: The parameters.
    :type params: dict
    :raises NotImplementedError: Missing parameter.
    :return: The parameters, corrected/updated where needed.
    :rtype: dict
    """

    # Throw an error if there's a missing parameter.
    if params["source"] == "":
        Utils.exception(
            'Missing parameter "source". You need to specify '
            + "the source of the input molecules (probably a SMI or SDF "
            + "file)."
        )

    # Note on parameter "source", the data source. If it's a string that
    # ends in ".smi", it's treated as a smiles file. If it's a string that
    # ends in ".sdf", it's treated as an sdf file. If it's any other
    # string, it's assumed to be a smiles string itself and is assigned a
    # name of "". If it's a list, it's assumed to be a list of tuples,
    # [SMILES, Name].

    # Check some required variables.
    try:
        params["source"] = os.path.abspath(params["source"])
    except:
        Utils.exception("Source file doesn't exist.")
    source_dir = params["source"].strip(os.path.basename(params["source"]))

    if params["output_folder"] == "" and params["source"] != "":
        params["output_folder"] = source_dir + "output" + str(os.sep)

    if params["add_pdb_output"] == True and params["output_folder"] == "":
        Utils.exception("To output files as .pdbs, specify the output_folder.")

    if params["separate_output_files"] == True and params["output_folder"] == "":
        Utils.exception("For separate_output_files, specify the output_folder.")

    # if not os.path.exists(params["output_folder"]) or not os.path.isdir(params["output_folder"]):
    #     Utils.exception(
    #         "The specified \"output_folder\", " + params["output_folder"] +
    #         ", either does not exist or is a file rather than a folder. " +
    #         "Please provide the path to an existing folder instead."
    #     )

    # Make sure job_manager is always lower case.
    params["job_manager"] = params["job_manager"].lower()

    return params


def add_mol_id_props(contnrs):
    """Once all molecules have been generated, go through each and add the
       name and a unique id (for writing to the SDF file, for example).

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    """

    cont_id = 0
    for contnr in contnrs:
        for mol in contnr.mols:
            cont_id = cont_id + 1
            mol.set_rdkit_mol_prop("UniqueID", str(cont_id))
            mol.set_all_rdkit_mol_props()


def deal_with_failed_molecules(contnrs, params):
    """Removes and logs failed molecules.

    :param contnrs: A list of containers (MolContainer.MolContainer).
    :type contnrs: list
    :param params: The parameters, used to determine the filename that will
       contain the failed molecules.
    :type params: dict
    """

    failed_ones = []  # To keep track of failed molecules
    for contnr in contnrs:
        if len(contnr.mols) == 0:
            astr = contnr.orig_smi + "\t" + contnr.name
            failed_ones.append(astr)

    # Let the user know if there's more than one failed molecule.
    if len(failed_ones) > 0:
        Utils.log("\n3D models could not be generated for the following entries:")
        Utils.log("\n".join(failed_ones))
        Utils.log("\n")

        # Write the failures to an smi file.
        outfile = open(params["output_folder"] + os.sep + "gypsum_dl_failed.smi", "w")
        outfile.write("\n".join(failed_ones))
        outfile.close()
