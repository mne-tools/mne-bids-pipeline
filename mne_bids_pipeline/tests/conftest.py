"""Pytest config."""


def pytest_addoption(parser):
    parser.addoption(
        "--download",
        action="store_true",
        help="Download data for selected tests to ~/mne_data.",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "dataset_test: mark that a test runs a dataset"
    )
