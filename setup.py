from setuptools import setup, find_packages

setup(
    name='MolScore',
    version='1.0',
    packages=['molscore'] + ['molscore.'+p for p in find_packages(where="molscore")] + ['moleval'] + ['moleval.'+p for p in find_packages(where="moleval")],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A scoring framework for goal directed generative models',
    include_package_data=True,
    package_data={
        'molscore': [
            'data/sample.smi',
            'data/models/RAScore/*/*.pkl',
            'data/models/pidgin/*'],
        'moleval': [
            'test/data/sample.smi',
            'metrics/mcf.csv',
            'metrics/wehi_pains.csv']},
)