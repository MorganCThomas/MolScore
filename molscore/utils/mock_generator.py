import random
from importlib import resources


class MockGenerator:
    """
    Mock generator that provides molecules as SMILES
    """

    def __init__(
        self,
        smi_file: str = None,
        seed_no: int = 123,
        augment_invalids: bool = False,
        augment_duplicates: bool = False,
    ):
        """
        Mock generator that provides molecules as SMILES given a smiles_file, otherwise, uses inbuilt sample
         from ChEMBL.

        :param smi_file: Path to smiles file (.smi)
        :param seed_no: Random seed
        :param augment_invalids: Forcefully augment invalid smiles by shuffling SMILES strings
        :param augment_duplicates: Forcefully augment duplicates by copying certain SMILES strings
        """
        if smi_file is None:
            with resources.open_text("molscore.data", "sample.smi") as f:
                self.smiles = f.read().splitlines()
        else:
            with open(smi_file, "r") as f:
                self.smiles = f.read().splitlines()
        self.augment_invalids = augment_invalids
        self.augment_duplicates = augment_duplicates
        random.seed(seed_no)

    def sample(self, size: int):
        """
        Sample SMILES
        :param size: Number of SMILES to sample
        :return: List of SMILES
        """
        sample_smiles = random.sample(self.smiles, size)
        duplicated_smiles = []
        invalid_smiles = []

        if self.augment_invalids or self.augment_duplicates:
            fraction_to_augment = 0.1
            sample_size = round(len(sample_smiles) * fraction_to_augment)

            if self.augment_duplicates:
                # Remove smiles randomly so that total number doesn't change
                for i in range(sample_size):
                    random.shuffle(sample_smiles)
                    sample_smiles.pop()

                # Randomly select and copy smiles, this way a smiles may be copied more than once
                for i in range(sample_size):
                    random_index = round(random.uniform(0, len(sample_smiles) - 1))
                    sample_smiles.append(sample_smiles[random_index])
                    duplicated_smiles.append(sample_smiles)

                # Give it a final shuffle
                random.shuffle(sample_smiles)

            if self.augment_invalids:
                # Manually sample so we can avoid sampling duplicated smiles
                i = 0
                while i < sample_size:
                    random_index = round(random.uniform(0, len(sample_smiles) - 1))
                    if len(duplicated_smiles) > 0:
                        if sample_smiles[random_index] not in duplicated_smiles:
                            invalid_smiles.append(
                                sample_smiles.pop(random_index)
                            )  # remove from sample size
                            i += 1
                    else:
                        invalid_smiles.append(sample_smiles.pop(random_index))
                        i += 1

                # Shuffle invalid smiles to make them hopefully invalid
                invalid_smiles = [list(smi) for smi in invalid_smiles]
                for i in range(len(invalid_smiles)):
                    random.shuffle(invalid_smiles[i])
                invalid_smiles = ["".join(smi) for smi in invalid_smiles]
                sample_smiles += invalid_smiles

                # Give it a final shuffle
                random.shuffle(sample_smiles)

        return sample_smiles
