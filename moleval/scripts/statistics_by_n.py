import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
import math

from moleval.metrics.metrics import GetMosesMetrics

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def calculate_n_statistics(results, train, test, test_scaffolds, target, ptrain, ptest, ptarget,
                           n_jobs, n_col=None, n=None):
    """
    Function that iterates of molscore results and summarises per K molecules.

    :param results: Dataframe of results from molscore.
    :param train: List of SMILES used for training.
    :param ptrain: FCD summary statistics of training smiles (.pkl).
    :return: Dataframe summarised per 1000 molecules.
    """

    n_metrics = []
    get_moses = GetMosesMetrics(n_jobs=n_jobs, device='cuda:1', batch_size=512,
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
            moses_metrics = get_moses.calculate(smiles, se_k=None)
            sum_metrics.update(moses_metrics)
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
            moses_metrics = get_moses.calculate(smiles, se_k=None)
            sum_metrics.update(moses_metrics)
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
            moses_metrics = get_moses.calculate(smiles, se_k=None)
            sum_metrics.update(moses_metrics)
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
    parser = argparse.ArgumentParser(description='Summarise molscore results per K molecules for ease of analysis.')
    parser.add_argument('-i', dest='input', help='Path to relevant scores.csv file.')
    parser.add_argument('--train', help='Smiles used for training (.smi).')
    parser.add_argument('--test', help='Smiles used for test (.smi).')
    parser.add_argument('--test_scaff', help='Smiles used for test scaffold (.smi).')
    parser.add_argument('--target', help='Smiles used for target (.smi)')
    parser.add_argument('--o', dest='output', help='Output directory to save output to.')
    parser.add_argument('--n_col', help='Split data by column definining step / iteration.', type=str)
    parser.add_argument('--n', help='Split data into groups of specified size.', type=int)
    parser.add_argument('--ptrain', help='FCD summary statistics for training data (.pkl).')
    parser.add_argument('--ptest', help='FCD summary statistics for test data (.pkl).')
    parser.add_argument('--ptarget', help='FCD summary statistics for target data - #5000 minimum ideally (.pkl).')
    parser.add_argument('--n_jobs', help='Number of jobs for parallel processing where possible.', default=1,
                        type=int)
    args = parser.parse_args()

    # Load in files
    if args.input:
        results = pd.read_csv(args.input, index_col=0, dtype={'valid': object,
                                                              'unique': object})

    if args.train:
        with open(args.train, 'rt') as f:
            train = f.read().splitlines()
    else:
        train = None

    if args.test:
        with open(args.test, 'rt') as f:
            test = f.read().splitlines()
    else:
        test = None

    if args.test_scaff:
        with open(args.test_scaff, 'rt') as f:
            test_scaff = f.read().splitlines()
    else:
        test_scaff = None

    if args.target:
        with open(args.target, 'rt') as f:
            target = f.read().splitlines()
    else:
        target = None

    if args.ptrain:
        with open(args.ptrain, 'rb') as f:
            ptrain = pkl.load(f)
    else:
        ptrain = None

    if args.ptest:
        with open(args.ptest, 'rb') as f:
            ptest = pkl.load(f)
    else:
        ptest = None

    if args.ptarget:
        with open(args.ptarget, 'rb') as f:
            ptarget = pkl.load(f)
    else:
        ptarget = None

    # Print basic summary
    print('Processing {}'.format(args.input))
    print('Size: {}\nValid: {}%\nUnique: {}%'.format(len(results),
                                                     round((results['valid'] == 'true').mean() * 100, 2),
                                                     round((results['unique'] == 'true').mean() * 100, 2)))

    # Calculate Statistics
    n_results = calculate_n_statistics(results=results,
                                       train=train, test=test, test_scaffolds=test_scaff, target=target,
                                       ptrain=ptrain, ptest=ptest, ptarget=ptarget,
                                       n_jobs=args.n_jobs,
                                       n_col=args.n_col, n=args.n)

    in_name = os.path.basename(args.input).split(".")[0]
    out_name = f"{in_name}_summary"
    # Check directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # If similar file, rename.
    if os.path.exists(os.path.join(args.output, f"{out_name}.csv")):
        print('Warning: Found pre-existing file that would be overwritten. Appending data and time.')
        ctime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        n_results.to_csv(os.path.join(args.output, f"{out_name}-{ctime}.csv"))

    else:
        n_results.to_csv(os.path.join(args.output, f"{out_name}.csv"))
