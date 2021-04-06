"""
Adapted from guacamol_baselines https://github.com/BenevolentAI/guacamol_baselines
"""

import argparse
import os

# from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
# from guacamol.utils.helpers import setup_default_logger

from directed_generator import SmilesRnnDirectedGenerator
from molscore.manager import MolScore

from torch import cuda
cuda.device(1)

def get_args():
    parser = argparse.ArgumentParser(description='Goal-directed generation benchmark for SMILES RNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore_config', '-m', type=str, help='Path to molscore config (.json)')
    parser.add_argument('--model_path', default=None, help='Full path to the pre-trained SMILES RNN model')
    parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--number_repetitions', default=1, type=int, help='Number of re-training runs to average')
    parser.add_argument('--keep_top', default=512, type=int, help='Molecules kept each step')
    parser.add_argument('--n_epochs', default=20, type=int, help='Epochs to sample')
    parser.add_argument('--mols_to_sample', default=1024, type=int, help='Molecules sampled at each step')
    parser.add_argument('--optimize_batch_size', default=256, type=int, help='Batch size for the optimization')
    parser.add_argument('--optimize_n_epochs', default=2, type=int, help='Number of epochs for the optimization')
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    return args


def main(args):
    ms = MolScore(model_name='lstm_hc', task_config=args.molscore_config)

    if args.model_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.model_path = os.path.join(dir_path, 'pretrained_model', 'model_final_0.473.pt')

    optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=args.model_path,
                                           n_epochs=args.n_epochs,
                                           mols_to_sample=args.mols_to_sample,
                                           keep_top=args.keep_top,
                                           optimize_n_epochs=args.optimize_n_epochs,
                                           max_len=args.max_len,
                                           optimize_batch_size=args.optimize_batch_size,
                                           random_start=args.random_start,
                                           smi_file=args.smiles_file,
                                           n_jobs=args.n_jobs)
    ms.log_parameters({'n_epochs': args.n_epochs,
                       'sample_size': args.mols_to_sample,
                       'keep_top': args.keep_top,
                       'optimize_n_epochs': args.optimize_n_epochs,
                       'batch_size': args.optimize_batch_size,
                       'pretrained_model': args.model_path,
                       'smi_file': args.smiles_file})

    final_population_smiles = optimizer.generate_optimized_molecules(scoring_function=ms,
                                                                     number_molecules=0)
    ms.write_scores()
    ms.kill_dash_monitor()

    with open(os.path.join(ms.save_dir, 'final_sample.smi'), 'w') as f:
        [f.write(smi + '\n') for smi in final_population_smiles]
    return


if __name__ == '__main__':
    # setup_default_logger()
    args = get_args()
    main(args)

