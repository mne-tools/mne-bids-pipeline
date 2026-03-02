"""Download test data."""

import argparse
from pathlib import Path
from warnings import filterwarnings

import mne

from ._config_import import _import_config
from ._config_utils import get_fs_subjects_dir
from .tests.datasets import DATASET_OPTIONS

DEFAULT_DATA_DIR = Path("~/mne_data").expanduser()


# TODO this can be removed when https://github.com/fatiando/pooch/pull/458 is merged and
# we pin to a version of pooch that includes that commit
filterwarnings(
    action="ignore",
    message=(
        "Python 3.14 will, by default, filter extracted tar archives and reject files "
        "or modify their metadata. Use the filter argument to control this behavior."
    ),
    category=DeprecationWarning,
    module="tarfile",
)


def _download_via_openneuro(*, ds_name: str, ds_path: Path) -> None:
    import openneuro

    options = DATASET_OPTIONS[ds_name]
    assert "hash" not in options

    kwargs = dict(
        dataset=options["openneuro"],
        target_dir=ds_path,
        include=options.get("include", []),
        exclude=options.get("exclude", []),
    )
    print(f"Downloading with openneuro.download(**{kwargs})")
    openneuro.download(**kwargs)


def _download_from_web(*, ds_name: str, ds_path: Path) -> None:
    """Retrieve `.zip` or `.tar.gz` archives from a web URL."""
    import pooch

    options = DATASET_OPTIONS[ds_name]
    url = options["web"]
    known_hash = options["hash"]
    assert "exclude" not in options
    assert "include" not in options
    if ds_path.exists():
        print(
            "Dataset directory already exists; remove it if you wish to "
            "re-download the dataset"
        )
        return

    ds_path.mkdir(parents=True, exist_ok=True)
    path = ds_path.parent.resolve(strict=True)
    ext = "tar.gz" if options.get("processor") == "untar" else "zip"
    processor = pooch.Untar if options.get("processor") == "untar" else pooch.Unzip
    fname = f"{ds_name}.{ext}"
    pooch.retrieve(
        url=url,
        path=path,
        fname=fname,
        processor=processor(extract_dir="."),  # relative to path
        progressbar=True,
        known_hash=known_hash,
    )
    (path / f"{ds_name}.{ext}").unlink()


def _download_via_mne(*, ds_name: str, ds_path: Path) -> None:
    assert ds_path.stem == ds_name, ds_path
    getattr(mne.datasets, DATASET_OPTIONS[ds_name]["mne"]).data_path(
        ds_path.parent,
        verbose=True,
    )


def _download(*, ds_name: str, ds_path: Path) -> None:
    options = DATASET_OPTIONS[ds_name]
    openneuro_name = options.get("openneuro", "")
    web_url = options.get("web", "")
    mne_mod = options.get("mne", "")
    assert sum(bool(x) for x in (openneuro_name, web_url, mne_mod)) == 1

    if openneuro_name:
        download_func = _download_via_openneuro
    elif mne_mod:
        download_func = _download_via_mne
    else:
        assert web_url
        download_func = _download_from_web

    download_func(ds_name=ds_name, ds_path=ds_path)

    # and fsaverage if needed
    extra = DATASET_OPTIONS[ds_name].get("config_path_extra", "")
    config_path = (
        Path(__file__).parent
        / "tests"
        / "configs"
        / f"config_{ds_name.replace('-', '_')}{extra}.py"
    )
    if config_path.is_file():
        has_subjects_dir = any(
            "derivatives/freesurfer/subjects" in key
            for key in options.get("include", [])
        )
        if has_subjects_dir or options.get("fsaverage"):
            cfg = _import_config(config_path=config_path)
            subjects_dir = get_fs_subjects_dir(config=cfg)
            n_try = 5
            for ii in range(1, n_try + 1):  # osf.io fails sometimes
                write_extra = f" (attempt #{ii})" if ii > 1 else ""
                print(f"Checking fsaverage in {subjects_dir} ...{write_extra}")
                try:
                    mne.datasets.fetch_fsaverage(
                        subjects_dir=subjects_dir,
                        verbose=True,
                    )
                except Exception:  # pragma: no cover
                    if ii == n_try:
                        raise
                    else:
                        print("Failed and will retry, got:\n{exc}")
                else:
                    break


def main(dataset: str | None) -> None:
    """Download the testing data."""
    # Save everything 'MNE_DATA' dir ... defaults to ~/mne_data
    mne_data_dir = mne.get_config(key="MNE_DATA", default=False)
    if not mne_data_dir:
        mne.set_config("MNE_DATA", str(DEFAULT_DATA_DIR))
        DEFAULT_DATA_DIR.mkdir(exist_ok=True)
        mne_data_dir = DEFAULT_DATA_DIR
    else:
        mne_data_dir = Path(mne_data_dir)

    ds_names = DATASET_OPTIONS.keys() if not dataset else (dataset,)

    for ds_name in ds_names:
        title = f"Downloading {ds_name}"
        bar = "-" * len(title)
        print(f"{title}\n{bar}")
        ds_path = mne_data_dir / ds_name
        _download(ds_name=ds_name, ds_path=ds_path)
        print()


if __name__ == "__main__":  # pragma: no cover
    # This is only used by CircleCI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="dataset",
        help="Name of the dataset",
        metavar="DATASET",
        nargs="?",
        default=None,
    )
    opt = parser.parse_args()
    dataset = opt.dataset if opt.dataset != "" else None
    main(dataset)
