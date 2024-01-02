import os
import json
import unittest
import subprocess

from molscore import MolScoreBenchmark
from molscore.tests.mock_generator import MockGenerator

def main():
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_out')
    MSB = MolScoreBenchmark(model_name='test', output_dir=output_directory, budget=10, benchmark='GuacaMol')
    mg = MockGenerator()
    for task in MSB:
        while not task.finished:
            smiles = mg.sample(5)
            task.score(smiles)

if __name__ == '__main__':
    main()