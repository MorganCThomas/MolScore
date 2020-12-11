from setuptools import setup

setup(
    name='molscore',
    version='1.0',
    packages=['molscore'],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A scoring framework for goal directed generative models',
    include_package_data=True,
    package_data={'molscore': ['test/data/sample.smi']},
)
