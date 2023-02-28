import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
import math

from moleval.metrics.metrics import GetMetrics
from moleval import utils

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')


def calculate_n_statistics(results, train, test, test_scaffolds, target, ptrain, ptest, ptarget,
                           n_jobs, n_col=None, n=None, device='cpu', run_fcd=False):
    """
    Function that iterates of molscore results and summarises per K molecules.

    :param results: Dataframe of results from molscore.
    :param train: List of SMILES used for training.
    :param ptrain: FCD summary statistics of training smiles (.pkl).
    :return: Dataframe summarised per 1000 molecules.
    """

    n_metrics = []
    get_metrics = GetMetrics(n_jobs=n_jobs, device=device, batch_size=512, run_fcd=run_fcd,
                             pool=None, train=train, test=test,
                             test_scaffolds=test_scaffolds, target=target,
                             ptest=ptest, ptrain=ptrain, ptarget=ptarget)

    if (n_col is not None) and (n is None):
        for i in tqdm(results[n_col].unique()):
            sum_metrics = {}
            sum_df = results[results[n_col] == i]
            smiles = sum_df['smiles'].unique().tolist()
            sum_metrics.update({'step': i})
            # ------ Compute Moses metrics ------
            moleval_metrics = get_metrics.calculate(smiles)
            sum_metrics.update(moleval_metrics)
            # ------ Compute molscore metrics ------
            sum_metrics.update({'Validity': (sum_df['valid'] == 'true').mean()})
            sum_metrics.update({'Uniqueness': (sum_df['unique'] == 'true').mean()})
            if 'passes_diversity_filter' in sum_df.columns:
                sum_metrics.update({'Passed_diversity_filter': (sum_df['passes_diversity_filter'] == 'true').mean()})
            for metric in sum_df.columns:
                # Anything that is a float, int besides batch_idx...
                if (any([isinstance(sum_df[metric].values[0], dtype) for dtype in [np.int64, np.float64]])) and \
                        (metric not in ['batch_idx']):
                    sum_metrics.update({'{}_mean'.format(metric): (sum_df[metric]).mean()})
                    sum_metrics.update({'{}_median'.format(metric): (sum_df[metric]).median()})
                    sum_metrics.update({'{}_std'.format(metric): (sum_df[metric]).std()})
            # Append to list per n.
            n_metrics.append(sum_metrics)

    elif (n_col is None) and (n is not None):
        for i in tqdm(range(n, len(results), n)):
            sum_metrics = {}
            sum_df = results[i-n:i]
            smiles = sum_df['smiles'].unique().tolist()
            sum_metrics.update({'n': i})
            # ------ Compute Moses metrics ------
            moleval_metrics = get_metrics.calculate(smiles)
            sum_metrics.update(moleval_metrics)
            # ------ Compute molscore metrics ------
            sum_metrics.update({'Validity': (sum_df['valid'] == 'true').mean()})
            sum_metrics.update({'Uniqueness': (sum_df['unique'] == 'true').mean()})
            if 'passes_diversity_filter' in sum_df.columns:
                sum_metrics.update({'Passed_diversity_filter': (sum_df['passes_diversity_filter'] == 'true').mean()})
            for metric in sum_df.columns:
                # Anything that is a float or int besides step, batch_idx, absolute_time
                if (any([isinstance(sum_df[metric].values[0], dtype) for dtype in [np.int64, np.float64]])) and \
                        (metric not in ['batch_idx', 'absolute_time']):
                    sum_metrics.update({'{}_mean'.format(metric): (sum_df[metric]).mean()})
                    sum_metrics.update({'{}_median'.format(metric): (sum_df[metric]).median()})
                    sum_metrics.update({'{}_std'.format(metric): (sum_df[metric]).std()})
            # Append to list per k.
            n_metrics.append(sum_metrics)
    else:
        for i in tqdm(range(results[n_col].values[0],
                            math.ceil(results[n_col].values[-1]/n)*n,  # original ... * n+1
                            n)):  # original range(n, ...)
            sum_metrics = {}
            sum_df = results[(results[n_col] >= i) & (results[n_col] < i+n)]  # original >= i-n
            smiles = sum_df['smiles'].unique().tolist()
            sum_metrics.update({'step': i+n})
            # ------ Compute Moses metrics ------
            moleval_metrics = get_metrics.calculate(smiles)
            sum_metrics.update(moleval_metrics)
            # ------ Compute molscore metrics ------
            sum_metrics.update({'Validity': (sum_df['valid'] == 'true').mean()})
            sum_metrics.update({'Uniqueness': (sum_df['unique'] == 'true').mean()})
            if 'passes_diversity_filter' in sum_df.columns:
                sum_metrics.update({'Passed_diversity_filter': (sum_df['passes_diversity_filter'] == 'true').mean()})
            for metric in sum_df.columns:
                # Anything that is a float, int besides batch_idx...
                if (any([isinstance(sum_df[metric].values[0], dtype) for dtype in [np.int64, np.float64]])) and \
                        (metric not in ['batch_idx']):
                    sum_metrics.update({'{}_mean'.format(metric): (sum_df[metric]).mean()})
                    sum_metrics.update({'{}_median'.format(metric): (sum_df[metric]).median()})
                    sum_metrics.update({'{}_std'.format(metric): (sum_df[metric]).std()})
            # Append to list per n.
            n_metrics.append(sum_metrics)

    return pd.DataFrame(n_metrics)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                                    """
                                    Summarise molscore results per n molecules or n values by calculating the mean, median and std as well as calculating moleval metrics. 
                                    For example, n=100 is every 100 molecules, n_col=step is every step, n=100 and n_col=step is every 100 steps.
                                    Either n or n_col must be provided.
                                    """
                                    )
    parser.add_argument('input', help='Path to relevant scores.csv file.')
    parser.add_argument('--output_dir', dest='output', help='Output directory to save output to, default is parent directory of input.')
    parser.add_argument('--n_col', help='Split data by column definining step / iteration.', type=str, default=None)
    parser.add_argument('--n', help='Split data into groups of specified size.', type=int, default=None)
    parser.add_argument('--n_jobs', help='Number of jobs for parallel processing where possible.', default=1, type=int)
    parser.add_argument('--run_fcd', action='store_true', help='Run Frechet Chemnet Distance calculations (FCD)')
    parser.add_argument('--cpu', action='store_true', help='Only use CPU during metric calculations')

    ref_sets = parser.add_argument_group('Reference datasets', 'Reference datasets to compare to de novo molecules. If not provided, will compute intrinsic properties only.')
    ref_sets.add_argument('--train', help='Smiles used for training (.smi).', default=None)
    ref_sets.add_argument('--test', help='Smiles used for test (.smi).', default=None)
    ref_sets.add_argument('--test_scaff', help='Smiles used for test scaffold (.smi).', default=None)
    ref_sets.add_argument('--target', help='Smiles used for target (.smi)', default=None)
    
    fcd_stats = parser.add_argument_group('FCD statistics', 'Pre-prepared FCD statistics for reference datasets')
    fcd_stats.add_argument('--ptrain', help='FCD summary statistics for training data (.pkl).', default=None)
    fcd_stats.add_argument('--ptest', help='FCD summary statistics for test data (.pkl).', default=None)
    fcd_stats.add_argument('--ptarget', help='FCD summary statistics for target data - #5000 minimum ideally (.pkl).', default=None)

    args = parser.parse_args()

    # ----- Prepare inputs -----
    # Load in scores.csv
    results = pd.read_csv(args.input, index_col=0, dtype={'valid': object, 'unique': object})
    # Check n/n_col
    assert (args.n is not None) or (args.n_col is not None), "Either \'n\' or \'n_col\' must be provided. Try \'--n 100\' maybe?"
    # Load in reference datasets
    if args.train:
        args.train = utils.read_smiles(args.train)
    if args.test:
        args.test = utils.read_smiles(args.test)
    if args.test_scaff:
        args.test_scaff = utils.read_smiles(args.test_scaff)
    if args.target:
        args.target = utils.read_smiles(args.target)
    # Load in pre-prepared FCD statistics
    if args.ptrain:
        args.ptrain = utils.read_pickle(args.ptrain)
    if args.ptest:
        args.ptest = utils.read_pickle(args.ptest)
    if args.ptarget:
        args.ptarget = utils.read_pickle(args.ptarget)
    # Set device
    if utils.cuda_available() and not (args.cpu):
        device = 'cuda:1'
    else:
        device = 'cpu'

    # ----- Basic summary -----
    print('Processing {}'.format(args.input))
    print('Size: {}\nValid: {}%\nUnique: {}%'.format(len(results),
                                                     round((results['valid'] == 'true').mean() * 100, 2),
                                                     round((results['unique'] == 'true').mean() * 100, 2)))

    # ----- Calculate Statistics -----
    n_results = calculate_n_statistics(results=results,
                                       train=args.train, test=args.test, test_scaffolds=args.test_scaff, target=args.target,
                                       ptrain=args.ptrain, ptest=args.ptest, ptarget=args.ptarget,
                                       n_jobs=args.n_jobs, n_col=args.n_col, n=args.n, device=device, run_fcd=args.run_fcd)
    
    # ----- Save output -----
    # Output file name
    in_name = os.path.basename(args.input).split(".")[0]
    out_name = f"{in_name}_summary"
    # Output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = os.path.dirname(args.input)
    # If similar file, rename.
    if os.path.exists(os.path.join(args.output, f"{out_name}.csv")):
        print('Warning: Found pre-existing file that would be overwritten. Appending data and time.')
        ctime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        n_results.to_csv(os.path.join(args.output, f"{out_name}-{ctime}.csv"))

    else:
        n_results.to_csv(os.path.join(args.output, f"{out_name}.csv"))
