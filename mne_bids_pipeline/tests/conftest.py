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
    warning_lines = r"""
    error::
    ignore:There is no current event loop:DeprecationWarning
    ignore:distutils Version classes are deprecated.*:DeprecationWarning
    ignore:The register_cmap function.*:PendingDeprecationWarning
    ignore:Jupyter is migrating its paths[.\n]*:DeprecationWarning
    ignore:numpy\.ndarray size changed, may indicate binary.*:RuntimeWarning
    always::ResourceWarning
    ignore:subprocess .* is still running:ResourceWarning
    ignore:`np.MachAr` is deprecated.*:DeprecationWarning
    ignore:The get_cmap function will be deprecated.*:
    ignore:make_current is deprecated.*:DeprecationWarning
    """
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)
