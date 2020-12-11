from setuptools import setup

setup(
    name='moleval',
    version='1.0',
    packages=['moleval'],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A evaluation framework for goal directed generative models',
    include_package_data=True,
    package_data={'moleval': ['test/data/sample.smi']}
)
