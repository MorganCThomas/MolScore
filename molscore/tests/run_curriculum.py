import argparse
import os

from molscore import MockGenerator, MolScoreCurriculum


def main(benchmark):
    output_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_out"
    )
    mg = MockGenerator(augment_invalids=True, augment_duplicates=True)

    # Use config budget of 50
    MSC = MolScoreCurriculum(
        model_name="test",
        output_dir=output_directory,
        custom_benchmark=benchmark,
        run_name="Curriculum",
    )
    while not MSC.finished:
        smiles = mg.sample(10)
        MSC.score(smiles)
    MSC.write_scores()
    print(f"Output:\n{MSC.main_df.head(10)}\n")

    # Specify a new budget of 100
    MSC = MolScoreCurriculum(
        model_name="test",
        output_dir=output_directory,
        custom_benchmark=benchmark,
        run_name="Curriculum",
        budget=100,
    )
    while not MSC.finished:
        smiles = mg.sample(10)
        MSC.score(smiles)
    MSC.write_scores()
    print(f"Output:\n{MSC.main_df.head(10)}\n")

    # Specify a budget of 100, termination_threshold of 0.8
    MSC = MolScoreCurriculum(
        model_name="test",
        output_dir=output_directory,
        custom_benchmark=benchmark,
        run_name="Curriculum",
        budget=100,
        termination_threshold=0.5,
    )
    while not MSC.finished:
        smiles = mg.sample(10)
        MSC.score(smiles)
    MSC.write_scores()
    print(f"Output:\n{MSC.main_df.head(10)}\n")

    # Specify a budget of 100, termination_threshold of 0.8, patience of 10
    MSC = MolScoreCurriculum(
        model_name="test",
        output_dir=output_directory,
        custom_benchmark=benchmark,
        run_name="Curriculum",
        budget=100,
        termination_threshold=0.5,
        termination_patience=5,
    )
    while not MSC.finished:
        smiles = mg.sample(10)
        MSC.score(smiles)
    MSC.write_scores()
    print(f"Output:\n{MSC.main_df.head(10)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str, help="Path to config directory")
    args = parser.parse_args()

    main(args.benchmark)
