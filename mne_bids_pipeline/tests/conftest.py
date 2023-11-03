"""Pytest config."""


def pytest_addoption(parser):
    parser.addoption(
        "--download",
        action="store_true",
        help="Download data for selected tests to ~/mne_data.",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line("markers", "dataset_test: mark that a test runs a dataset")
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
    ignore:`np.*` is a deprecated alias for .*:DeprecationWarning
    ignore:.*implicit namespace.*:DeprecationWarning
    ignore:Deprecated call to `pkg_resources.*:DeprecationWarning
    ignore:.*declare_namespace.*mpl_toolkits.*:DeprecationWarning
    ignore:_SixMetaPathImporter\.find_spec.*:ImportWarning
    ignore:pkg_resources is deprecated.*:DeprecationWarning
    ignore:`product` is deprecated as of NumPy.*:DeprecationWarning
    # seaborn calling tight layout, etc.
    ignore:The figure layout has changed to tight:UserWarning
    ignore:The \S+_cmap function was deprecated.*:DeprecationWarning
    # seaborn->pandas
    ignore:is_categorical_dtype is deprecated.*:FutureWarning
    ignore:use_inf_as_na option is deprecated.*:FutureWarning
    # Dask distributed with jsonschema 4.18
    ignore:jsonschema\.RefResolver is deprecated.*:DeprecationWarning
    # seaborn->pandas
    ignore:is_categorical_dtype is deprecated.*:FutureWarning
    ignore:use_inf_as_na option is deprecated.*:FutureWarning
    ignore:All-NaN axis encountered.*:RuntimeWarning
    # sklearn class not enough samples for cv=5
    always:The least populated class in y has only.*:UserWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
