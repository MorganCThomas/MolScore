from setuptools import setup, find_packages

setup(
    name='moleval',
    version='1.0',
    packages=['moleval'] + ['moleval.'+p for p in find_packages(where="moleval")],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A evaluation framework for goal directed generative models',
    include_package_data=True,
    package_data={'moleval': ['test/data/sample.smi', 'metrics/mcf.csv', 'metrics/wehi_pains.csv']}
)
