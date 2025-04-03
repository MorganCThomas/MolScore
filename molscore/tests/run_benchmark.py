import argparse
import os

from tqdm import tqdm

from molscore import MockGenerator, MolScoreBenchmark


def main(benchmark):
    output_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_out"
    )
    MSB = MolScoreBenchmark(
        model_name="test", output_dir=output_directory, budget=10, benchmark=benchmark
    )
    with MSB as benchmark:
        mg = MockGenerator(augment_invalids=True, augment_duplicates=True)
        for task in tqdm(benchmark, desc="Benchmark Objectives"):
            with task as scoring_function:
                while not scoring_function.finished:
                    smiles = mg.sample(10)
                    scoring_function.score(smiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        type=str,
        default="GuacaMol",
        choices=[
            "GuacaMol",
            "MolOpt",
            "MolExp",
            "MolExpL",
            "MolExp_baseline",
            "MolExpL_baseline",
            "MolOpt-CF",
            "MolOpt-DF",
            "5HT2A_PhysChem",
            "5HT2A_Selectivity",
            "5HT2A_Docking",
        ],
    )
    args = parser.parse_args()

    main(args.benchmark)
