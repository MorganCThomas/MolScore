"""
Adapted from guacamol_baselines https://github.com/BenevolentAI/guacamol_baselines
"""

import logging
import time
from functools import total_ordering
from typing import List, Set
from joblib import delayed

import numpy as np
import torch
import torch.nn as nn

#from guacamol.scoring_function import ScoringFunction
from utils import canonicalize_list

from model import SmilesRnn
from sampler import SmilesRnnSampler
from trainer import SmilesRnnTrainer
from utils import get_tensor_dataset, load_smiles_from_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score

    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)

    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)


class SmilesRnnMoleculeGenerator:
    """
    character-based RNN language model optimized by with hill-climbing

    """

    def __init__(self, model: SmilesRnn, max_len: int, device: str) -> None:
        """
        Args:
            model: Pre-trained SmilesRnn
            max_len: maximum SMILES length
            device: 'cpu' | 'cuda'
        """

        self.device = device
        self.model = model

        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.sampler = SmilesRnnSampler(device=self.device, batch_size=512)
        self.max_len = max_len
        self.trainer = SmilesRnnTrainer(model=self.model,
                                        criteria=[self.criterion],
                                        optimizer=self.optimizer,
                                        device=self.device)

    def optimise(self, objective, start_population, keep_top, n_epochs, mols_to_sample,
                 optimize_n_epochs, optimize_batch_size, pretrain_n_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param objective: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param kwargs need to contain:
                keep_top: number of molecules to keep at each iterative finetune step
                mols_to_sample: number of molecules to sample at each iterative finetune step
                optimize_n_epochs: number of episodes to finetune
                optimize_batch_size: batch size for fine-tuning
                pretrain_n_epochs: number of epochs to pretrain on start population
        :return: Candidate molecules
        """

        int_results = self.pretrain_on_initial_population(objective, start_population,
                                                          pretrain_epochs=pretrain_n_epochs)

        results: List[OptResult] = []
        seen: Set[str] = set()

        for k in int_results:
            if k.smiles not in seen:
                results.append(k)
                seen.add(k.smiles)

        for epoch in range(1, 1 + n_epochs):

            t0 = time.time()
            samples = self.sampler.sample(self.model, mols_to_sample, max_seq_len=self.max_len)
            t1 = time.time()

            canonicalized_samples = set(canonicalize_list(samples, include_stereocenters=True))
            payload = list(canonicalized_samples.difference(seen))
            payload.sort()  # necessary for reproducibility between different runs

            seen.update(canonicalized_samples)

            scores = objective(payload)
            int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(payload, scores)]

            t2 = time.time()

            results.extend(sorted(int_results, reverse=True)[0:keep_top])
            results.sort(reverse=True)
            subset = [i.smiles for i in results][0:keep_top]

            np.random.shuffle(subset)

            sub_train = subset[0:int(3 * len(subset) / 4)]
            sub_test = subset[int(3 * len(subset) / 4):]

            train_seqs, _ = load_smiles_from_list(sub_train, max_len=self.max_len)
            valid_seqs, _ = load_smiles_from_list(sub_test, max_len=self.max_len)

            train_set = get_tensor_dataset(train_seqs)
            valid_set = get_tensor_dataset(valid_seqs)

            opt_batch_size = min(len(sub_train), optimize_batch_size)

            print_every = int(len(sub_train) / opt_batch_size)

            if optimize_n_epochs > 0:
                self.trainer.fit(train_set, valid_set,
                                 n_epochs=optimize_n_epochs,
                                 batch_size=opt_batch_size,
                                 print_every=print_every,
                                 valid_every=print_every)

            t3 = time.time()

            logger.info(f'Generation {epoch} --- timings: '
                        f'sample: {(t1 - t0):.3f} s, '
                        f'score: {(t2 - t1):.3f} s, '
                        f'finetune: {(t3 - t2):.3f} s')

            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in results[:4])
            logger.info(f'Top 4:\n{top4}')

        return sorted(results, reverse=True)

    def sample(self, num_mols) -> List[str]:
        """

        :return: a list of molecules
        """

        return self.sampler.sample(self.model,
                                   num_to_sample=num_mols,
                                   max_seq_len=self.max_len)

    # TODO refactor, still has lots of duplication
    def pretrain_on_initial_population(self, scoring_function,
                                       start_population, pretrain_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param scoring_function: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param pretrain_epochs: number of epochs to finetune with start_population
        :return: Candidate molecules
        """
        seed: List[OptResult] = []

        start_population_size = len(start_population)

        training = canonicalize_list(start_population, include_stereocenters=True)

        if len(training) != start_population_size:
            logger.warning("Some entries for the start population are invalid or duplicated")
            start_population_size = len(training)

        if start_population_size == 0:
            return seed

        logger.info("finetuning with {} molecules for {} epochs".format(start_population_size, pretrain_epochs))

        scores = scoring_function(training, flt=True, score_only=True)
        seed.extend(OptResult(smiles=smiles, score=score) for smiles, score in zip(training, scores))

        train_seqs, _ = load_smiles_from_list(training, max_len=self.max_len)
        train_set = get_tensor_dataset(train_seqs)

        batch_size = min(int(len(training)), 32)

        print_every = len(training) / batch_size

        losses = self.trainer.fit(train_set, train_set,
                                  batch_size=batch_size,
                                  n_epochs=pretrain_epochs,
                                  print_every=print_every,
                                  valid_every=print_every)
        logger.info(losses)
        return seed
