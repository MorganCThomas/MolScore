import json
import os
import subprocess
import unittest

from molscore.scoring_functions import (
    ASPGOLDDock,
    ChemPLPGOLDDock,
    ChemScoreGOLDDock,
    GlideDock,
    GninaDock,
    GOLDDock,
    GoldScoreGOLDDock,
    PLANTSDock,
    SminaDock,
    VinaDock,
    rDock,
)
from molscore.tests import BaseTests, MockGenerator, test_files


class TestGlideDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Check installation
        if "SCHRODINGER" not in list(os.environ.keys()):
            raise unittest.SkipTest("Schrodinger installation not found")
        # Check license
        license_check = (
            subprocess.run(
                "$SCHRODINGER/licadmin STAT",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            .stdout.decode()
            .split("\n")
        )
        for line in license_check:
            if "Error getting status:" in line:
                raise unittest.SkipTest(line)

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Prepare a grid file
        input_file = os.path.join(cls.output_directory, "glide.in")
        with open(input_file, "w") as f:
            f.write(f"GRIDFILE   {test_files['GlideDock_grid']}\n")
            f.write("PRECISION    SP\n")
        # Instantiate
        cls.obj = GlideDock
        cls.inst = GlideDock(
            prefix="test",
            glide_template=input_file,
            ligand_preparation="LigPrep",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGlideDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGlideDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Check installation
        if "SCHRODINGER" not in list(os.environ.keys()):
            raise unittest.SkipTest("Schrodinger installation not found")
        # Check license
        license_check = (
            subprocess.run(
                "$SCHRODINGER/licadmin STAT",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            .stdout.decode()
            .split("\n")
        )
        for line in license_check:
            if "Error getting status:" in line:
                raise unittest.SkipTest(line)

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Prepare a grid file
        input_file = os.path.join(cls.output_directory, "glide.in")
        with open(input_file, "w") as f:
            f.write(f"GRIDFILE   {test_files['GlideDock_grid']}\n")
            f.write("PRECISION    SP\n")
        # Instantiate
        cls.obj = GlideDock
        cls.inst = GlideDock(
            prefix="test",
            glide_template=input_file,
            ligand_preparation="LigPrep",
            cluster=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGlideDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestSminaDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = SminaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Smina installation not found")
        # Instantiate
        cls.inst = SminaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor_pdbqt"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cpus=8,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nSminaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestSminaDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = SminaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Smina installation not found")
        # Instantiate
        cls.inst = SminaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor_pdbqt"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cpus=1,
            ligand_preparation="GypsumDL",
            cluster=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nSminaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGninaDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = GninaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Gnina installation not found")
        # Instantiate
        cls.inst = GninaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor_pdbqt"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cpus=8,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=True,
        )
        print(f"\nGninaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGninaDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = GninaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Gnina installation not found")
        # Instantiate
        cls.inst = GninaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor_pdbqt"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cpus=1,
            ligand_preparation="GypsumDL",
            cluster=1,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=True,
        )
        print(f"\nGninaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestVinaDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = VinaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Vina installation not found")
        # Instantiate
        cls.inst = VinaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            file_preparation="mgltools",
            cpus=2,
            ligand_preparation="GypsumDL",
            dock_scoring="vina",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=False,
        )
        print(f"\nVinaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestVinaDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        cls.obj = VinaDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("Vina installation not found")
        # Instantiate
        cls.inst = VinaDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            file_preparation="mgltools",
            cpus=1,
            cluster=4,
            ligand_preparation="GypsumDL",
            dock_scoring="vina",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=False,
        )
        print(f"\nVinaDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestPLANTSDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = PLANTSDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("PLANTS installation not found")
        # Instantiate
        cls.inst = PLANTSDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nPLANTSDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestPLANTSDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = PLANTSDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("PLANTS installation not found")
        # Instantiate
        cls.inst = PLANTSDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nPLANTSDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGOLDDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = GOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = GOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGOLDDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = GOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = GOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestChemPLPGOLDDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = ChemPLPGOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = ChemPLPGOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestASPGOLDDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = ASPGOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = ASPGOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestChemScoreGOLDDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = ChemScoreGOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = ChemScoreGOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGoldScoreGOLDDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = GoldScoreGOLDDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("GOLD installation not found")
        # Instantiate
        cls.inst = GoldScoreGOLDDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            cluster=4,
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nGOLDDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestOEDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Check installation
        try:
            from molscore.scoring_functions.oedock import OEDock
        except ImportError:
            raise unittest.SkipTest(
                "OpenEye not found, please install via mamba install openeye-toolkits -c openeye"
            )
        if "OE_LICENSE" not in list(os.environ.keys()):
            raise unittest.SkipTest(
                "OpenEye license not found, please install license and export to 'OE_LICENSE'"
            )

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = OEDock
        cls.inst = OEDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nOEDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestOEDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Check installation
        try:
            from molscore.scoring_functions.oedock import OEDock
        except ImportError:
            raise unittest.SkipTest(
                "OpenEye not found, please install via mamba install openeye-toolkits -c openeye"
            )
        if "OE_LICENSE" not in list(os.environ.keys()):
            raise unittest.SkipTest(
                "OpenEye license not found, please install license and export to 'OE_LICENSE'"
            )

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = OEDock
        cls.inst = OEDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            cluster=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nOEDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestrDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = rDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("rDock installation not found")
        # Instantiate
        cls.inst = rDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            n_runs=2,
            cluster=1,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nrDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestrDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = rDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("rDock installation not found")
        # Instantiate
        cls.inst = rDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            n_runs=2,
            cluster=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nrDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestrDockParallelPH4(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = rDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("rDock installation not found")
        # Instantiate
        cls.inst = rDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            dock_constraints=test_files["DRD2_rdock_constraint"],
            n_runs=2,
            cluster=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [test_files["DRD2_ref_smiles"]] + mg.sample(4)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=True,
        )
        print(f"\nrDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestrDockParallelScaff1(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = rDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("rDock installation not found")
        # Instantiate
        cls.inst = rDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            dock_substructure_constraints="Fc1cc2oncc2cc1",
            dock_substructure_max_trans=0.0,
            dock_substructure_max_rot=0.0,
            n_runs=5,
            cluster=4,
        )
        # Call
        cls.input = [test_files["DRD2_ref_smiles"]] + [
            "Fc1cc2onc(CC)c2cc1",
            "Fc1cc2onc(CCC(=O))c2cc1",
            "Fc1cc2onc(C(=O)N)c2cc1",
            "Fc1cc2onc(CCCCC)c2cc1",
        ]
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=True,
        )
        print(f"\nrDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestrDockParallelScaff2(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)

        cls.obj = rDock
        # Check installation
        try:
            cls.obj.check_installation()
        except RuntimeError:
            raise unittest.SkipTest("rDock installation not found")
        # Instantiate
        cls.inst = rDock(
            prefix="test",
            receptor=test_files["DRD2_receptor"],
            ref_ligand=test_files["DRD2_ref_ligand"],
            ligand_preparation="GypsumDL",
            dock_substructure_constraints="Fc1cc2oncc2cc1.Cc1nc2CCCCn2c(=O)c1",
            dock_substructure_max_trans=0.0,
            dock_substructure_max_rot=0.0,
            n_runs=5,
            cluster=4,
        )
        # Call
        cls.input = [test_files["DRD2_ref_smiles"]] + [
            "Cc1nc2CCCCn2c(=O)c1CCCCCc1noc2cc(F)ccc21",
            "Cc1nc2CCCCn2c(=O)c1CCCCc1noc2cc(F)ccc21",
            "Cc1nc2CCCCn2c(=O)c1CCCCCCc1noc2cc(F)ccc21",
            "Cc1nc2CCCCn2c(=O)c1CCOCCc1noc2cc(F)ccc21",
        ]
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input,
            directory=cls.output_directory,
            file_names=file_names,
            cleanup=True,
        )
        print(f"\nrDock Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst.client.close()
        cls.inst.client.cluster.close()
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


if __name__ == "__main__":
    unittest.main()
