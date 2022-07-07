#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import argparse
import os
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import decrease_learning_rate
rdBase.DisableLog('rdApp.error')


def pretrain(smi_file, voc_file, output_dir, suffix, n_epochs=5, restore_from=None, device=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)

    # Create a Dataset from a SMILES file
    moldata = MolData(smi_file, voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc, device)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, n_epochs+1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), os.path.join(output_dir, f"Prior_{suffix}.ckpt"))

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), os.path.join(output_dir, f"Prior_{suffix}.ckpt"))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to smiles file (.smi)'
    )
    parser.add_argument(
        '--voc', '-v',
        type=str,
        help='Path to Vocabulary file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output directory'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        help='Suffix used to name files e.g. noDRD2'
    )
    parser.add_argument(
        '-d', '--device',
        type=int,
        default=None,
        help='GPU Device'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        help='Number of training epochs',
        default=5
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pretrain(smi_file=args.input, voc_file=args.voc, output_dir=args.output,
             n_epochs=args.n_epochs, suffix=args.suffix, device=args.device)
