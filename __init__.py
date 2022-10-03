from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mne_bids_pipeline")
except PackageNotFoundError:
    # package is not installed
    pass
