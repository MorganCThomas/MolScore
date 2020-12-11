import torch
import os
from tqdm import tqdm
from model import RNN
import argparse
from data_structs import Vocabulary
from utils import seq_to_smiles, unique


def main(voc_file='data/Voc',
         restore_model_from='data/Prior.ckpt',
         output_file='data/Prior_10k.smi',
         sample_size=10000):

    voc = Vocabulary(init_from_file=voc_file)
    print("Setting up networks")
    Agent = RNN(voc)

    if torch.cuda.is_available():
        print("Cuda available, loading prior & agent")
        Agent.rnn.load_state_dict(torch.load(restore_model_from))
    else:
        raise 'Cuda not available'


    SMILES = []
    for n in tqdm(range(sample_size//100), total=sample_size//100):
        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(100)
        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]
        smiles = seq_to_smiles(seqs, voc)
        SMILES += smiles

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "wt") as f:
        [f.write(smi + '\n') for smi in SMILES]

    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--voc', '-v',
        type=str,
        help='Path to Vocabulary file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save file e.g. Data/Prior_10k.smi)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=10000,
        help='Number of molecules to sample (default is 10,0000)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(restore_model_from=args.model, voc_file=args.voc,
         output_file=args.output, sample_size=args.sample_size)
