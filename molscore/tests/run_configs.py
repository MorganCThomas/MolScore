import os
import sys
from glob import glob

from tqdm import tqdm

from molscore import MockGenerator, MolScore


def main(configs: list):
    mg = MockGenerator(augment_invalids=True, augment_duplicates=True)
    for config in configs:
        print(f"\nTesting: {config}")
        # Run
        print(f"Running {os.path.basename(config).split('.')[0]}:")
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=os.path.join(os.path.dirname(__file__), "test_out"),
        )
        # Score 5 smiles 5 times
        for i in tqdm(range(5)):
            _ = ms(mg.sample(10))
        ms.write_scores()
        ms.kill_monitor()
        print(f"Output:\n{ms.main_df.head(10)}\n")
    return


if __name__ == "__main__":
    configs = []
    if len(sys.argv) == 1:
        print("No config files or directories of config files provided")
    else:
        for arg in sys.argv[1:]:
            if not os.path.exists(arg):
                print(f"{arg} does not exist")
            else:
                if os.path.isdir(arg):
                    configs.extend(glob(os.path.join(arg, "*.json")))
                else:
                    configs.append(arg)
    main(configs)
