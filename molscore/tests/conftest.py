def pytest_addoption(parser):
    parser.addoption(
        "--configs",
        action="store",
        nargs="+",
        help="One or more configuration definitions, paths, or directories.",
    )