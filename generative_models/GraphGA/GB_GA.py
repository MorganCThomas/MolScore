'''
Adapted from:
  Written by Jan H. Jensen 2018.
  Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
And:
  https://github.com/BenevolentAI/guacamol_baselines/blob/jtvae/graph_ga/goal_directed_generation.py
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol

from rdkit import rdBase

rdBase.DisableLog('rdApp.*')

from typing import List, Optional
import numpy as np
import argparse
import random
from time import time
# import heapq
# import sys
import os
import joblib
from joblib import delayed

import crossover as co
import mutate as mu

from molscore.manager import MolScore
# import scoring_functions as sc


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


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def sanitize(population_mol):
    """

    :param population_mol:
    :return:
    """
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population


class GB_GA:

    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, n_jobs=-1,
                 random_start=False, patience=5):
        self.smi_file = smi_file
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(self.canonicalize)(s.strip()) for s in f)

    @staticmethod
    def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
        else:
            return None

    def top_k(self, smiles, scoring_function, k):
        scores = scoring_function(smiles, flt=True, score_only=True)
        # joblist = (delayed(scoring_function.score)(s) for s in smiles)
        # scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        # fetch initial population?
        if starting_population is None:
            if self.random_start:
                print('Randomly selecting initial population...')
                starting_population = np.random.choice(self.all_smiles, self.population_size)
            else:
                print('Selecting highest scoring initial population...')
                starting_population = self.top_k(self.all_smiles, scoring_function, self.population_size)

        # select initial population
        starting_population_scores = scoring_function(starting_population, flt=True, score_only=True)
        population_smiles = []
        population_mol = []
        population_scores = []
        for score, smi in sorted(zip(starting_population_scores, starting_population),
                                 key=lambda x: x[0], reverse=True)[:self.population_size]:
            population_smiles.append(smi)
            population_mol.append(Chem.MolFromSmiles(smi))
            population_scores.append(score)

        # evolution: go go go!!
        t0 = time()

        patience = 0

        for generation in range(self.generations):

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = self.pool(
                delayed(reproduce)(mating_pool, self.mutation_rate) for _ in range(self.population_size))
            print(f'Returning {len([m for m in offspring_mol if m is not None])} mutants')

            # add new_population
            population_mol += sanitize(offspring_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            population_scores = scoring_function([Chem.MolToSmiles(s) for s in population_mol],
                                                 step=generation,
                                                 flt=True)
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')

        # finally
        return [Chem.MolToSmiles(m) for m in population_mol] # [:number_molecules]


def main(args):
    ms = MolScore(model_name='graphGA', task_config=args.molscore_config)
    generator = GB_GA(smi_file=args.smiles_file,
                      population_size=args.population_size,
                      offspring_size=args.offspring_size,
                      generations=args.generations,
                      mutation_rate=args.mutation_rate,
                      n_jobs=args.n_jobs,
                      random_start=args.random_start,
                      patience=args.patience)
    ms.log_parameters({'population_size': args.population_size,
                       'offspring_size': args.offspring_size,
                       'mutation_rate': args.mutation_rate,
                       'patience': args.patience,
                       'random_start': args.random_start})
    final_population_smiles = generator.generate_optimized_molecules(scoring_function=ms)
    ms.write_scores()
    ms.kill_dash_monitor()

    with open(os.path.join(ms.save_dir, 'final_population.smi'), 'w') as f:
        [f.write(smi + '\n') for smi in final_population_smiles]
    return


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore_config', '-m', type=str, help='Path to molscore config (.json)')
    parser.add_argument('--smiles_file', type=str, help=' ')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--seed', type=int, default=0, help=' ')
    optional.add_argument('--population_size', type=int, default=1000, help=' ')
    optional.add_argument('--offspring_size', type=int, default=200, help=' ')
    optional.add_argument('--mutation_rate', type=float, default=0.01, help=' ')
    optional.add_argument('--generations', type=int, default=50, help=' ')
    optional.add_argument('--n_jobs', type=int, default=-1, help=' ')
    optional.add_argument('--random_start', action='store_true')
    optional.add_argument('--patience', type=int, default=5, help=' ')
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
