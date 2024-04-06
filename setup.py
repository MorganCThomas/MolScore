from setuptools import setup, find_packages

setup(
    packages=['molscore'] + ['molscore.'+p for p in find_packages(where="molscore")] + ['moleval'] + ['moleval.'+p for p in find_packages(where="moleval")],
    include_package_data=True,
    scripts=['molscore/gui/config.py', 'molscore/gui/molscore_config', 
             'molscore/gui/monitor.py', 'molscore/gui/molscore_monitor'],
    package_data={
        'molscore': [
            'data/sample.smi',
            'data/sample.bloom',
            'data/models/**/*',
            'data/structures/**/*',
            'configs/**/*',
            'utils/gypsum_dl/**/*'],
        'moleval': [
            'test/data/sample.smi',
            'metrics/mcf.csv',
            'metrics/wehi_pains.csv',
            'metrics/quality_filters/alert_collection.csv',
            'metrics/quality_filters/guacamol_rules.json']},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "rdkit >= 2019.03.2, < 2024", # 2024 changes MolStandardize import
        "dask", # == 2023.6.0",
        "dask-jobqueue", # == 0.8.2",
        "scikit-learn == 1.1.3",
        "torch",
        "zenodo-client",
        "levenshtein",
        "streamlit-plotly-events == 0.0.6",
        "streamlit == 1.29.0",
        "molbloom",
        "func_timeout",
        "flask"
    ],
)