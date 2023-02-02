import argparse
import os
import pickle as pkl
from moleval.metrics.fcd_torch import FCD
from moleval.utils import disable_rdkit_log

disable_rdkit_log()


def fcd_statistics(SMILES, n_jobs, gpu, out, canonicalize=False):
    """
    Pre-compute FCD statistics for a list of SMILES.

    :param SMILES: List SMILES
    :param n_jobs: Number of jobs for processing SMILES
    :param out: Path to output file
    :return:
    """
    print('Building model')
    fcd = FCD(device=f'cuda:{gpu}', n_jobs=n_jobs, canonize=canonicalize)

    print('Calculating FCD')
    results = fcd.precalc(SMILES)

    print('Saving pre-statistics')
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    with open(out, 'wb') as f:
        pkl.dump(results, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pre-compute FCD statistics for a list of SMILES and pickle results.')
    parser.add_argument('-i, --input', dest='input', help='List of smiles in a file (.smi).')
    parser.add_argument('-o, --output', dest='output', help='Output file name.')
    parser.add_argument('--n_jobs', dest='n_jobs', help='Number of jobs for parallel processing of SMILES.',
                        default=1, type=int)
    parser.add_argument('--gpu', dest='gpu_device', help='GPU Device (default 0)', type=int, default=0)
    parser.add_argument('--can', action='store_true',
                        default=False,
                        help='Whether to canonicalize smiles with rdkit')
    args = parser.parse_args()

    # Process smiles
    print('Loading smiles')
    with open(args.input, 'r') as f:
        SMILES = f.read().splitlines()

    fcd_statistics(SMILES=SMILES, n_jobs=args.n_jobs, gpu=args.gpu_device, out=args.output,
                   canonicalize=args.can)

