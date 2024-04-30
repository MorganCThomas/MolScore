import argparse
import os

from tqdm import tqdm

from molscore import MockGenerator, MolScoreCurriculum


def main(benchmark):
    output_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_out"
    )
    MSC = MolScoreCurriculum(
        model_name="test", output_dir=output_directory, custom_benchmark=benchmark
    )
    mg = MockGenerator(augment_invalids=True, augment_duplicates=True)
    while not MSC.finished:
        smiles = mg.sample(10)
        MSC.score(smiles)
    MSC.write_scores()
    print(f"Output:\n{MSC.main_df.head(10)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        type=str,
        help="Path to config directory"
    )
    args = parser.parse_args()

    main(args.benchmark)
