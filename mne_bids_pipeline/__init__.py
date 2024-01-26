from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mne_bids_pipeline")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0"
