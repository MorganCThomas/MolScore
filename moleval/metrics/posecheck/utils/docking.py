import os
from typing import Union

import numpy as np
from rdkit import Chem

from .constants import BOX_SIZE, SMINA_PATH
from .docking_utils import *
from .loading import read_sdf

# TODO add this to .env file
SMINA_PATH = "smina"


class SMINA(object):
    def __init__(
        self,
        executable=SMINA_PATH,
        cpu=None,
    ):
        """Wrapper class for smina docking software.

        Args:
            executable (str): Path to smina executable.
            cpu (int, optional): Number of CPU cores to use. If None, all available cores will be used (recommended).
        """
        self.executable = executable
        self.minimize = False
        self.score_only = False
        self.cpu = cpu

    def set_receptor(
        self,
        receptor_path: Union[str, os.PathLike],
        centre: Union[str, Chem.Mol, np.ndarray] = "pocket",
        size: int = BOX_SIZE,
    ) -> None:
        """
        Set the receptor for docking.

        Args:
            receptor_path (Union[str, os.PathLike]): Path to the receptor file.
            centre (Union[str, Chem.Mol, np.ndarray]): The centre of the docking box.
                If 'pocket', the centre is calculated from the PDBQT file.
                If Chem.Mol, the centre is calculated from the RDKit molecule.
                If np.ndarray, the centre is already specified as a numpy array.
                If str, the centre is read from an SDF file.
            size (int): The size of the docking box. Default is 25.

        Raises:
            ValueError: If the centre type is not one of the specified types.
        """

        self.receptor = receptor_path

        if centre == "pocket":
            # Get the receptor centre from the PDBQT file with rdkit
            receptor_mol = read_pdbqt_receptor(receptor_path)
            centre = get_centroid_ob(receptor_mol)
        elif type(centre) == Chem.Mol:
            # Get the receptor centre from the rdkit mol
            centre = get_centroid_rdmol(centre)
        elif type(centre) == np.ndarray:
            # Centre is already a numpy array
            pass
        elif type(centre) == str:  # TODO add support for PDBQT
            # get from SDF
            try:
                mol = read_sdf(centre)[0]
                centre = get_centroid_rdmol(mol)
            except:
                raise ValueError(
                    f"Could not read centre from file {centre}. Make sure it is a valid SDF file."
                )
        else:
            raise ValueError(
                f'Invalid centre type {type(centre)}. Must be one of: "pocket", Chem.Mol, np.ndarray, or path to sdf.'
            )

        # Set the centre and size of the docking box
        self.centre_x = centre[0]
        self.centre_y = centre[1]
        self.centre_z = centre[2]

        self.size_x = size
        self.size_y = size
        self.size_z = size

    def set_ligand(self, ligand_path):
        raise NotImplementedError(
            "Use set_ligand_from_mol instead, centering is not implemented for this method."
        )
        # self.ligand = ligand_path

    def set_ligand_from_mol(self, mol: Chem.Mol):
        self.task_id = np.random.randint(0, 1000000)
        self.ligand = f"tmp_{self.task_id}.pdbqt"

        rdkit_mol_to_pdbqt(mol, self.ligand)

    def clear_ligand(self):
        """Remove the ligand file. Important to do this after each docking run."""
        if os.path.exists(self.ligand):
            os.remove(self.ligand)

    def score_pose(self):
        self.score_only = True
        mol, out = self.run()
        self.score_only = False

        out = parse_smina_output_score(out)

        if len(mol) == 1:
            out["mol"] = mol[0]
        else:
            out["mol"] = None

        return out

    def minimize_pose(self):
        self.minimize = True
        mol, out = self.run()
        self.minimize = False

        out = parse_smina_output_minimize(out)

        if len(mol) == 1:
            out["mol"] = mol[0]
        else:
            out["mol"] = None

        return out

    def redock(self):
        mols, out = self.run()
        out = parse_smina_output_docking(out)

        out["mols"] = mols

        return out

    def calculate_all(self, mol=None):
        if mol is not None:
            self.set_ligand_from_mol(mol)

        try:
            score_only = self.score_pose()
            minimized = self.minimize_pose()
            redocked = self.redock()

            self.clear_ligand()

            return {
                "score_only": score_only,
                "minimized": minimized,
                "redocked": redocked,
            }

        except:
            self.clear_ligand()
            return {"score_only": None, "minimized": None, "redocked": None}

    def run(self):
        """Main function to run smina.
        Should be called from one of the other functions.
        """

        tmp_out = f"tmp_out_{self.task_id}.pdbqt"

        # Command over multiple lines for readability
        command = (
            f"{self.executable} -r {self.receptor} -l {self.ligand} "
            f"--center_x {self.centre_x} --center_y {self.centre_y} --center_z {self.centre_z} "
            f"--size_x {self.size_x} --size_y {self.size_y} --size_z {self.size_z} "
            f"-o {tmp_out} "
        )

        if not self.minimize and not self.score_only:
            command += "--exhaustiveness 8"

        if self.minimize:
            command += "--minimize "
        if self.score_only:
            command += "--score_only "

        if self.cpu:
            command += f"--cpu {self.cpu} "  # only do if you want use less than the max number of cores on your machine

        # Perform docking/minimization
        # print(command)
        # os.system(command)
        out = os.popen(command).read()

        out_mols = pdbqt_to_rdkit_mols(tmp_out)
        os.remove(tmp_out)

        return out_mols, out
