from setuptools import setup, find_packages

setup(
    name='MolScore',
    version='1.1',
    packages=['molscore'] + ['molscore.'+p for p in find_packages(where="molscore")] + ['moleval'] + ['moleval.'+p for p in find_packages(where="moleval")],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A scoring framework for goal directed generative models',
    include_package_data=True,
    scripts=['molscore/gui/config.py', 'molscore/gui/molscore_config', 
             'molscore/gui/monitor.py', 'molscore/gui/molscore_monitor'],
    package_data={
        'molscore': [
            'data/sample.smi',
            'data/models/RAScore/*/*.pkl',
            'data/models/aizynth/*',
            'data/models/libinvent/*',
            'data/models/molopt/*'
            'configs/*/*',
            'configs/*/*/*',],
        'moleval': [
            'test/data/sample.smi',
            'metrics/mcf.csv',
            'metrics/wehi_pains.csv']},
)