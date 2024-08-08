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
This module is for testing Gypsum-DL. Not quite unit tests, but good enough
for now.
"""

import os
import shutil
import glob
from gypsum_dl import Utils
from gypsum_dl.Start import prepare_molecules


def run_test():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = script_dir + os.sep + "gypsum_dl_test_output" + os.sep

    # Delete test output directory if it exists.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Make the directory
    os.mkdir(output_folder)

    # Make the Gypsum-DL parameters.
    params = {
        "source": script_dir + os.sep + "sample_molecules.smi",
        "separate_output_files": True,
        "job_manager": "serial",  # multiprocessing
        "output_folder": output_folder,
        "add_pdb_output": False,
        "max_variants_per_compound": 8,
        "thoroughness": 1,
        "min_ph": 4,
        "max_ph": 10,
        "pka_precision": 1,
        "use_durrant_lab_filters": True,
    }

    # Prepare the molecules.
    prepare_molecules(params)
    Utils.log("")
    Utils.log("TEST RESULTS")
    Utils.log("============")

    # Get the output sdf files.
    sdf_files = glob.glob(output_folder + "*")

    # There should be seven sdf files.
    msg = "Expected 15 output files, got " + str(len(sdf_files)) + "."
    if len(sdf_files) != 15:
        Utils.exception("FAILED. " + msg)
    else:
        Utils.log("PASSED. " + msg)

    # Get all the smiles from the files.
    all_smiles = set([])
    for sdf_file in sdf_files:
        lines = open(sdf_file).readlines()
        for i, line in enumerate(lines):
            if "<SMILES>" in line:
                all_smiles.add(lines[i + 1].strip())

    # List what the smiles should be.
    target_smiles = set([])

    # salt_and_ionization should produce two models (ionized and
    # deionized).
    target_smiles |= set(["[O-]c1ccccc1", "Oc1ccccc1"])

    # tautomer_and_cis_trans should produce three models (two tautomers, one
    # of them with alternate cis/trans).
    target_smiles |= set([r"C/C=C\O", "C/C=C/O", "CCC=O"])

    # two_chiral_one_unspecified_and_tautomer should produce four models.
    target_smiles |= set(
        [
            "CC(C)C(=O)[C@@](F)(Cl)C[C@@](C)(F)Cl",
            "CC(C)=C(O)[C@@](F)(Cl)C[C@@](C)(F)Cl",
            "CC(C)C(=O)[C@](F)(Cl)C[C@@](C)(F)Cl",
            "CC(C)=C(O)[C@](F)(Cl)C[C@@](C)(F)Cl",
        ]
    )

    # two_double_bonds_one_chiral_center should produce eight models.
    target_smiles |= set(
        [
            r"CC/C(C[C@@](C)(Cl)I)=C(I)\C(F)=C(/C)Cl",
            "CC/C(C[C@](C)(Cl)I)=C(I)/C(F)=C(/C)Cl",
            r"CC/C(C[C@](C)(Cl)I)=C(I)/C(F)=C(\C)Cl",
            r"CC/C(C[C@](C)(Cl)I)=C(I)\C(F)=C(\C)Cl",
            r"CC/C(C[C@@](C)(Cl)I)=C(I)/C(F)=C(\C)Cl",
            r"CC/C(C[C@@](C)(Cl)I)=C(I)\C(F)=C(\C)Cl",
            "CC/C(C[C@@](C)(Cl)I)=C(I)/C(F)=C(/C)Cl",
            r"CC/C(C[C@](C)(Cl)I)=C(I)\C(F)=C(/C)Cl",
        ]
    )

    # two_double_bonds_one_unspecified should produce two models.
    target_smiles |= set(
        [r"CC/C(C)=C(\Cl)C/C(I)=C(\C)F", r"CC/C(C)=C(/Cl)C/C(I)=C(\C)F"]
    )

    # non_aromatic_ring should produce one model. It will list it several
    # times, because different ring conformations of the same model.
    target_smiles |= set(["CC(C)(C)[C@H]1CC[C@@H](C(C)(C)C)CC1"])

    # There should be no =[N-] if Durrant lab filters are turned on. Note:
    # Removed "CC(=N)O" from below list because durrant lab filters now remove
    # iminols.
    target_smiles |= set(["CC([NH-])=O", "CC(N)=O"])

    # There should be no [N-]C=[N+] (CC(=O)[N-]C=[N+](C)C).
    target_smiles |= set(
        [
            r"C/C(O)=N\C=[N+](C)C",
            r"CC(=O)/N=C\[NH+](C)C",
            "CC(=O)/N=C/[NH+](C)C",
            "CC(=O)NC=[N+](C)C",
            "C/C(O)=N/C=[N+](C)C",
        ]
    )

    # There should be no [nH+]c[n-] (c1c[nH+]c[n-]1)
    target_smiles |= set(["c1c[n-]cn1", "c1c[nH+]c[nH]1", "c1c[nH]cn1"])

    # There should be no [#7+]~[#7+] (c1cc[nH+][nH+]c1)
    target_smiles |= set(["c1ccnnc1", "c1cc[nH+]nc1"])

    # There should be no [#7-]~[#7-] (CC(=O)[N-][N-]C(C)=O). Note that some
    # are commented out because Python2 and Python3 given different SMILES
    # strings that are all valid. See below to see how things are
    # consolodated. (Really this was probably a bad example to pick because
    # there are so many forms...)
    target_smiles |= set(
        [
            "CC(=O)NNC(C)=O",
            # r"CC(=O)N/N=C(\C)O",
            # r"CC(=O)[N-]/N=C(/C)O",
            # r"C/C(O)=N/N=C(\C)O",
            r"C/C(O)=N\N=C(/C)O",
            # r"CC(=O)[N-]/N=C(\C)O",
            # "CC(=O)[N-]NC(C)=O",
            # "CC(=O)N/N=C(/C)O"
        ]
    )

    # There should be no [!#7]~[#7+]~[#7-]~[!#7] (c1c[n-][nH+]c1)
    target_smiles |= set(["c1cn[n-]c1", "c1cn[nH]c1", "c1c[nH][nH+]c1"])

    # Azides can have adjacent +/- nitrogens.
    target_smiles |= set(["CN=[N+]=[N-]", "CN=[N+]=N"])

    # msg = "Expected " + str(len(target_smiles)) + " total SMILES, got " + \
    #     str(len(all_smiles)) + "."
    # if len(all_smiles) != len(target_smiles):
    #     Utils.exception("FAILED. " + msg)
    # else:
    #     Utils.log("PASSED. " + msg)

    # Python3 gives some smiles that are different than thsoe obtain with
    # Python2. But they are just different representations of the same thing.
    # Let's make the switch to the Python2 form for this test.
    all_smiles = set(["CN=[N+]=N" if s == "[H]N=[N+]=NC" else s for s in all_smiles])

    # Note: Commented out below because durrant lab filters now remove
    # iminols.
    # all_smiles = set(
    #     ["CC(=N)O" if s in [r"[H]/N=C(\C)O", "[H]/N=C(/C)O"] else s for s in all_smiles]
    # )

    all_smiles = set(
        [
            r"C/C(O)=N\N=C(/C)O"
            if s == r"C/C(O)=N/N=C(/C)O"
            else s  # Different one that turns up sometimes
            for s in all_smiles
        ]
    )
    all_smiles = set(
        [
            r"CC(=O)NNC(C)=O"
            if s
            in [
                r"CC(=O)[N-]/N=C(\C)O",
                r"C/C(O)=N/N=C(\C)O",
                r"CC(=O)N/N=C(\C)O",
                r"CC(=O)[N-]/N=C(/C)O",
                r"CC(=O)[N-]NC(C)=O",
                r"CC(=O)N/N=C(/C)O",
            ]
            else s  # Different one that turns up sometimes
            for s in all_smiles
        ]
    )

    if len(all_smiles ^ target_smiles) > 0:
        print(all_smiles)
        print(target_smiles)
        import pdb; pdb.set_trace()

        Utils.exception(
            "FAILED. "
            + "Got some SMILES I didn't expect (either in output or target list): "
            + " ".join(list(all_smiles ^ target_smiles))
        )
    else:
        Utils.log("PASSED. Gypsum-DL output the very SMILES strings I was expecting.")

    Utils.log("")

    # Delete test output directory if it exists.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
