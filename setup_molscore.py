from setuptools import setup, find_packages

setup(
    name='molscore',
    version='1.0',
    packages=['molscore'] + ['molscore.'+p for p in find_packages(where="molscore")],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A scoring framework for goal directed generative models',
    include_package_data=True,
    package_data={'molscore': ['data/sample.smi',
                               'data/models/RAScore/*/*.pkl',
                               'data/models/pidgin/*']},
)
