import os
import sys
import json
from tqdm import tqdm
from glob import glob

from molscore.tests import MockGenerator
from molscore.manager import MolScore


def main(configs: list):
    mg = MockGenerator()
    for config in configs:
        # Ensure output directory is correct
        with open(config, 'rt') as f:
            cf = json.load(f)
        cf['output_dir'] = os.path.join(os.path.dirname(__file__), 'test_out')
        with open(config, 'wt') as f:
            json.dump(cf, f, indent=2)
        # Run
        print(f"Running {os.path.basename(config).split('.')[0]}:")
        ms = MolScore(model_name='test', task_config=config)
        # Score 5 smiles 5 times
        for i in tqdm(range(5)):
            ms(mg.sample(5))
        ms.write_scores()
        ms.kill_monitor()
    return


if __name__ == '__main__':
    if len(sys.argv) == 1:
        os.chdir(os.path.join(os.path.dirname(__file__), 'configs'))
        configs = glob('*.json')
    else:
        configs = sys.argv[1:]
    main(configs)