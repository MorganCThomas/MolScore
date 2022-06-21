from optimizer import BasePSOptimizer
from cddd.inference import InferenceModel
import argparse
import os
from molscore.manager import MolScore


def main(args):
    # Initialize molscore
    ms = MolScore(model_name='mso', task_config=args.molscore_config)
    ms.log_parameters(vars(args))

    # Load inference model
    infer_model = InferenceModel(model_dir=args.infer_model)

    # Load init smiles
    assert os.path.exists(args.smiles_file)
    with open(args.smiles_file, 'r') as f:
        init_smiles = f.read().splitlines()

    # Initialize Optimizer
    opt = BasePSOptimizer.from_query(
        init_smiles=init_smiles,
        num_part=args.num_part,
        num_swarms=args.num_swarm,
        inference_model=infer_model,
        scoring_function=ms)

    # Run optimizer
    opt.run(args.num_steps)

    # Save everything
    ms.write_scores()
    ms.kill_dash_monitor()


def get_args():
    parser = argparse.ArgumentParser(description='Goal-directed generation benchmark for SMILES RNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore_config', '-m', type=str, help='Path to molscore config (.json)')
    parser.add_argument('--infer_model', default=None, help='Full path to the pre-trained model'
                                                            ' directory (with hparams inside)')
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles',
                        help='Initial SMILES to start optimization from')
    parser.add_argument('--num_part', default=200, help='Number of swarm particles (if smiles file,'
                                                        ' this number of molecules will be randomly drawn)')
    parser.add_argument('--num_swarm', default=1, help='Number of swarms')
    parser.add_argument('--num_steps', default=200, help='Number of optimization steps')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
