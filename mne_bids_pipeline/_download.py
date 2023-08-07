"""Download test data."""
import argparse
from pathlib import Path

import mne

from .tests.datasets import DATASET_OPTIONS

DEFAULT_DATA_DIR = Path("~/mne_data").expanduser()


def _download_via_datalad(*, ds_name: str, ds_path: Path):
    import datalad.api as dl

    print('datalad installing "{}"'.format(ds_name))
    options = DATASET_OPTIONS[ds_name]
    git_url = options["git"]
    assert "exclude" not in options
    assert "hash" not in options
    dataset = dl.install(path=ds_path, source=git_url)

    # XXX: git-annex bug:
    # https://github.com/datalad/datalad/issues/3583
    # if datalad fails, use "get" twice, or set `n_jobs=1`
    if ds_name == "ds003104":
        n_jobs = 16
    else:
        n_jobs = 1

    for to_get in DATASET_OPTIONS[ds_name].get("include", []):
        print('datalad get data "{}" for "{}"'.format(to_get, ds_name))
        dataset.get(to_get, jobs=n_jobs)


def _download_via_openneuro(*, ds_name: str, ds_path: Path):
    import openneuro

    options = DATASET_OPTIONS[ds_name]
    assert "hash" not in options

    openneuro.download(
        dataset=options["openneuro"],
        target_dir=ds_path,
        include=options.get("include", []),
        exclude=options.get("exclude", []),
        verify_size=False,
    )


def _download_from_web(*, ds_name: str, ds_path: Path):
    """Retrieve Zip archives from a web URL."""
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
    fname = f"{ds_name}.zip"
    pooch.retrieve(
        url=url,
        path=path,
        fname=fname,
        processor=pooch.Unzip(extract_dir="."),  # relative to path
        progressbar=True,
        known_hash=known_hash,
    )
    (path / f"{ds_name}.zip").unlink()


def _download(*, ds_name: str, ds_path: Path):
    options = DATASET_OPTIONS[ds_name]
    openneuro_name = options.get("openneuro", "")
    git_url = options.get("git", "")
    osf_node = options.get("osf", "")
    web_url = options.get("web", "")
    assert sum(bool(x) for x in (openneuro_name, git_url, osf_node, web_url)) == 1

    if openneuro_name:
        download_func = _download_via_openneuro
    elif git_url:
        download_func = _download_via_datalad
    elif osf_node:
        raise RuntimeError("OSF downloads are currently not supported.")
    else:
        assert web_url
        download_func = _download_from_web

    download_func(ds_name=ds_name, ds_path=ds_path)


def main(dataset):
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
