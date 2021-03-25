#!/usr/bin/env python3
""" Adapted from
https://github.com/DrrDom/crem
https://doi.org/10.1186/s13321-020-00431-w
"""
#==============================================================================
# author          : Pavel Polishchuk
# date            : 26-06-2019
# version         : 
# python_version  : 
# copyright       : Pavel Polishchuk 2019
# license         : 
#==============================================================================

import argparse
import json
import os
import sys
from time import time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from crem import mutate_mol2
from molscore.manager import MolScore


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


class CREM_Generator:

    def __init__(self, smi_file, selection_size, db_fname, radius,
                 replacements, max_size, min_size, max_inc, min_inc,
                 generations, ncpu, random_start):
        self.pool = joblib.Parallel(n_jobs=ncpu)
        self.smiles = self.load_smiles_from_file(smi_file)
        self.N = selection_size
        self.db_fname = db_fname
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
        self.min_inc = min_inc
        self.max_inc = max_inc
        self.replacements = replacements
        self.replacements_baseline = replacements
        self.generations = generations
        self.random_start = random_start
        self.patience1 = 3
        self.patience2 = 10
        self.patience3 = 33
        self.task = 0

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return list(line.strip() for line in f)
            # return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        scores = scoring_function(smiles, flt=True, score_only=True)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate(self, smiles):
        mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
        res = self.pool(delayed(mutate_mol2)(mol, db_name=self.db_fname,
                                             radius=self.radius, min_size=self.min_size,
                                             max_size=self.max_size,
                                             min_rel_size=0, max_rel_size=1,
                                             min_inc=self.min_inc, max_inc=self.max_inc,
                                             max_replacements=self.replacements,
                                             replace_cycles=False,
                                             protected_ids=None, min_freq=0,
                                             return_rxn=False, return_rxn_freq=False,
                                             ncores=1) for mol in mols)
        res = set(m for sublist in res for m in sublist)
        return list(res)

    def set_params(self, score):
        # get min_inc, max_inc, max_replacements
        self.replacements = self.replacements_baseline
        if score > 0.8:
            self.min_inc = -4
            self.max_inc = 4
        elif score > 0.7:
            self.min_inc = -5
            self.max_inc = 5
        elif score > 0.6:
            self.min_inc = -6
            self.max_inc = 6
        elif score > 0.5:
            self.min_inc = -7
            self.max_inc = 7
        elif score > 0.4:
            self.min_inc = -8
            self.max_inc = 8
        elif score > 0.3:
            self.min_inc = -9
            self.max_inc = 9
        else:
            self.min_inc = -10
            self.max_inc = 10

    def generate_optimized_molecules(self, scoring_function,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        self.task += 1

        # select initial population
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                population = pd.DataFrame(np.random.choice(self.smiles, self.N), columns=['smi'])
            else:
                population = pd.DataFrame(self.top_k(self.smiles, scoring_function, self.N), columns=['smi'])
        else:
            population = pd.DataFrame(starting_population, columns=['smi'])
        population['score'] = scoring_function(population['smi'], flt=True)

        # evolution: go go go!!
        t0 = time()
        time_start = t0

        patience1 = 0
        patience2 = 0
        patience3 = 0

        best = population.copy().drop_duplicates(subset='smi')
        ref_score = np.mean(best['score'].iloc[:self.N])
        self.set_params(max(best['score']))
        used_smiles = set(population['smi'])   # smiles already used for mutation

        for generation in range(self.generations):

            if ref_score == 1:
                break

            population = pd.DataFrame(list(set(self.generate(population['smi']))), columns=['smi'])
            population['score'] = scoring_function(population['smi'], flt=True)
            population.sort_values(by='score', ascending=False, inplace=True)
            population.drop_duplicates(subset='smi', inplace=True)

            best = best.append(population).\
                drop_duplicates(subset='smi').\
                sort_values(by='score', ascending=False).\
                head(self.N)
            cur_score = np.mean(best['score'].iloc[:self.N])

            if cur_score > ref_score:
                ref_score = cur_score
                population = population.head(self.N)
                self.set_params(max(population['score']))
                used_smiles.update(population['smi'])
                patience1 = 0
                patience2 = 0
                patience3 = 0
            else:
                patience1 += 1
                patience2 += 1
                patience3 += 1
                if patience3 >= self.patience3:
                    if starting_population is None and self.random_start:
                        patience1 = 0
                        patience2 = 0
                        patience3 = 0
                        population = pd.DataFrame(np.random.choice(self.smiles, self.N), columns=['smi']).drop_duplicates(subset='smi')
                        population['score'] = scoring_function(population['smi'], flt=True)
                        population.sort_values(by='score', ascending=False, inplace=True)
                        self.set_params(max(population['score']))
                        used_smiles = set(population['smi'])
                    else:
                        break
                else:
                    population = population.head(self.N)
                    used_smiles.update(population['smi'])
                    if patience2 >= self.patience2:
                        patience1 = 0
                        patience2 = 0
                        self.min_inc -= 10
                        self.max_inc += 10
                        self.replacements += 500
                        used_smiles = set(population['smi'])
                    elif patience1 >= self.patience1:
                        patience1 = 0
                        self.min_inc -= 1
                        self.max_inc += 1
                        self.replacements += 100
                        used_smiles = set(population['smi'])

            # stats
            gen_time = time() - t0
            t0 = time()
            print(f'{generation: >5} | '
                  f'best avg: {np.round(np.mean(best["score"].iloc[:self.N]), 3)} | '
                  f'max: {np.max(population["score"]):.3f} | '
                  f'avg: {np.mean(population["score"]):.3f} | '
                  f'min: {np.min(population["score"]):.3f} | '
                  f'std: {np.std(population["score"]):.3f} | '
                  f'sum: {np.sum(population["score"]):.3f} | '
                  f'min_inc: {self.min_inc} | '
                  f'max_inc: {self.max_inc} | '
                  f'repl: {self.replacements} | '
                  f'p1: {patience1} | '
                  f'p2: {patience2} | '
                  f'p3: {patience3} | '
                  f'{gen_time:.2f} sec')
            sys.stdout.flush()

            if t0 - time_start > 18000:   # execution time > 5hr
                break

        # finally
        best.round({'score': 3}).to_csv(os.path.join(scoring_function.save_dir, f'{self.task}.smi'),
                                        sep="\t", header=False, index=False)
        return best['smi'][:self.N]


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore_config', type=str)
    parser.add_argument('--smiles_file', type=str)
    parser.add_argument('--db_fname', type=str)

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--selection_size', type=int, default=10, help=' ')
    optional.add_argument('--radius', type=int, default=3, help=' ')
    optional.add_argument('--replacements', type=int, default=1000, help=' ')
    optional.add_argument('--min_size', type=int, default=0, help=' ')
    optional.add_argument('--max_size', type=int, default=10, help=' ')
    optional.add_argument('--min_inc', type=int, default=-7, help=' ')
    optional.add_argument('--max_inc', type=int, default=7, help=' ')
    optional.add_argument('--generations', type=int, default=1000, help=' ')
    optional.add_argument('--ncpu', type=int, default=1, help=' ')
    optional.add_argument('--seed', type=int, default=42, help=' ')

    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


def main(args):
    scoring_function = MolScore(model_name='CReM', task_config=args.molscore_config)
    scoring_function.log_parameters({
        'selection_size': args.selection_size,
        'radius': args.radius,
        'n_generations': args.generations,
        'random_start': args.random_start
    })
    optimiser = CREM_Generator(smi_file=args.smiles_file,
                               selection_size=args.selection_size,
                               db_fname=args.db_fname,
                               radius=args.radius,
                               min_size=args.min_size,
                               max_size=args.max_size,
                               min_inc=args.min_inc,
                               max_inc=args.max_inc,
                               replacements=args.replacements,
                               generations=args.generations,
                               ncpu=args.ncpu,
                               random_start=True,)
    best = optimiser.generate_optimized_molecules(scoring_function=scoring_function)

    scoring_function.write_scores()
    scoring_function.kill_dash_monitor()

    with open(os.path.join(scoring_function.save_dir, 'best.smi'), 'w') as f:
        [f.write(s + '\n') for s in best]

    return


if __name__ == "__main__":
    args = get_args()
    main(args)
