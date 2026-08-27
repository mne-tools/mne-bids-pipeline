"""Pytest config."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest command line options."""
    parser.addoption(
        "--download",
        action="store_true",
        help="Download data for selected tests to ~/mne_data.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Add pytest configuration settings."""
    # register an additional marker
    config.addinivalue_line("markers", "dataset_test: mark that a test runs a dataset")
    warning_lines = r"""
    error::
    ignore:There is no current event loop:DeprecationWarning
    ignore:Jupyter is migrating its paths[.\n]*:DeprecationWarning
    ignore:numpy\.ndarray size changed, may indicate binary.*:RuntimeWarning
    always::ResourceWarning
    ignore:subprocess .* is still running:ResourceWarning
    ignore:.*implicit namespace.*:DeprecationWarning
    ignore:Deprecated call to `pkg_resources.*:DeprecationWarning
    ignore:pkg_resources is deprecated.*:DeprecationWarning
    # seaborn calling tight layout, etc.
    ignore:The figure layout has changed to tight:UserWarning
    # Dask distributed with jsonschema 4.18
    ignore:jsonschema\.RefResolver is deprecated.*:DeprecationWarning
    ignore:.*apply_async.*is deprecated.*:DeprecationWarning
    ignore:All-NaN axis encountered.*:RuntimeWarning
    # sklearn class not enough samples for cv=5
    always:The least populated class in y has only.*:UserWarning
    # constrained layout fails on ds003392
    # mne_bids_pipeline/steps/preprocessing/_06a_run_ica.py:551: in run_ica
    #    report.add_ica(
    #../python_env/lib/python3.10/site-packages/mne/report/report.py:1974: in add_ica
    #    self._add_ica(
    #../python_env/lib/python3.10/site-packages/mne/report/report.py:1872: in _add_ica
    #    self._add_ica_artifact_sources(
    #../python_env/lib/python3.10/site-packages/mne/report/report.py:1713:
    #  in _add_ica_artifact_sources
    #    self._add_figure(
    always:constrained_layout not applied.*:UserWarning
    ignore:.*ing of figures upon backend switching.*:
    ignore:datetime\.datetime\.utcfromtimestamp.*:DeprecationWarning
    ignore:datetime\.datetime\.utcnow.*:DeprecationWarning
    # pandas with no good workaround
    ignore:The behavior of DataFrame concatenation with empty.*:FutureWarning
    # joblib on Windows sometimes
    ignore:Persisting input arguments took.*:UserWarning
    # loky raises this in its manager thread, where an error kills the thread
    # and deadlocks the pool, so it must stay a warning
    always:A worker was restarted while some jobs were given.*:UserWarning
    # matplotlib needs to update
    ignore:Conversion of an array with ndim.*:DeprecationWarning
    # scipy
    ignore:nperseg .* is greater.*:UserWarning
    # NumPy 2.0
    ignore:__array_wrap__ must accept context.*:DeprecationWarning
    ignore:__array__ implementation doesn't accept.*:DeprecationWarning
    # Seaborn
    ignore:.*bool was deprecated in Matplotlib.*:DeprecationWarning
    ignore:.*bool will be deprecated.*:PendingDeprecationWarning
    # sklearn
    ignore:.*Liblinear failed to converge.*:
    # json-tricks
    ignore:json-tricks.*numpy scalar serialization.*:UserWarning
    # vtk<->NumPy 2.5
    ignore:.*Setting the shape on a NumPy array[\S\s]*:DeprecationWarning
    # meegkit
    ignore:pyriemann\.utils\.mean is deprecated.*:DeprecationWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
