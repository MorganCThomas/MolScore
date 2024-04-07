# Testing MolScore
Some unittests are available in the `tests directory` and can be run in the following way.

    cd molscore/tests
    python -m unittest

Or any individual test at a time, for example,

    python test_docking.py

Note that any scoring functions that spin up a flask server cannot be run with unittest which does not properly invoke the server. Therefore you must use [pytest](https://docs.pytest.org/en/8.0.x/) to test these features. Once installed it can be run via the following command. Note that pytest is directly compatible with running unittests too, if preferred... nice.

    pytest pytest_servers.py

Although no officially 'tests', you can run any configuration file to test it runs and check the output, for example,

    python run_configs.py <path_to_config1> <path_to_config2> <path_to_dir_of_configs>

Additionally, if you wish to run a preset benchmark you can run the following file.

    python run_benchmark.py GuacaMol

Run with `python run_benchmark.py --help` to see available preset benchmarks.