"""Download test data."""
import argparse
from pathlib import Path

import mne

from .tests.datasets import DATASET_OPTIONS

DEFAULT_DATA_DIR = Path('~/mne_data').expanduser()


def _download_via_datalad(*, ds_name: str, ds_path: Path):
    import datalad.api as dl

    print('datalad installing "{}"'.format(ds_name))
    git_url = DATASET_OPTIONS[ds_name]['git']
    dataset = dl.install(path=ds_path, source=git_url)

    # XXX: git-annex bug:
    # https://github.com/datalad/datalad/issues/3583
    # if datalad fails, use "get" twice, or set `n_jobs=1`
    if ds_name == 'ds003104':
        n_jobs = 16
    else:
        n_jobs = 1

    for to_get in DATASET_OPTIONS[ds_name]['include']:
        print('datalad get data "{}" for "{}"'.format(to_get, ds_name))
        dataset.get(to_get, jobs=n_jobs)


def _download_via_openneuro(*, ds_name: str, ds_path: Path):
    import openneuro
    openneuro.download(
        dataset=DATASET_OPTIONS[ds_name]['openneuro'],
        target_dir=ds_path,
        include=DATASET_OPTIONS[ds_name]['include'],
        exclude=DATASET_OPTIONS[ds_name]['exclude'],
        verify_size=False
    )


def _download_from_web(*, ds_name: str, ds_path: Path):
    """Retrieve Zip archives from a web URL.
    """
    import cgi
    import zipfile
    import httpx
    from tqdm import tqdm

    url = DATASET_OPTIONS[ds_name]['web']
    if ds_path.exists():
        print('Dataset directory already exists; remove it if you wish to '
              're-download the dataset')
        return

    ds_path.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client:
        with client.stream('GET', url=url) as response:
            if not response.is_error:
                pass  # All good!
            else:
                raise RuntimeError(
                    f'Error {response.status_code} when trying '
                    f'to download {url}')

            header = response.headers['content-disposition']
            _, params = cgi.parse_header(header)
            # where to store the archive
            outfile = ds_path / params['filename']
            remote_file_size = int(response.headers['content-length'])

            with open(outfile, mode='wb') as f:
                with tqdm(desc=params['filename'], initial=0,
                          total=remote_file_size, unit='B',
                          unit_scale=True, unit_divisor=1024,
                          leave=False) as progress:
                    num_bytes_downloaded = response.num_bytes_downloaded

                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        progress.update(response.num_bytes_downloaded -
                                        num_bytes_downloaded)
                        num_bytes_downloaded = (response
                                                .num_bytes_downloaded)

        assert outfile.suffix == '.zip'

        with zipfile.ZipFile(outfile) as zip:
            for zip_info in zip.infolist():
                path_in_zip = Path(zip_info.filename)
                # omit top-level directory from Zip archive
                target_path = str(Path(*path_in_zip.parts[1:]))
                if str(target_path) in ('.', '..'):
                    continue
                if zip_info.filename.endswith('/'):
                    (ds_path / target_path).mkdir(parents=True, exist_ok=True)
                    continue
                zip_info.filename = target_path
                print(f'Extracting: {target_path}')
                zip.extract(zip_info, ds_path)

        outfile.unlink()


def _download(*, ds_name: str, ds_path: Path):
    openneuro_name = DATASET_OPTIONS[ds_name]['openneuro']
    git_url = DATASET_OPTIONS[ds_name]['git']
    osf_node = DATASET_OPTIONS[ds_name]['osf']
    web_url = DATASET_OPTIONS[ds_name]['web']

    if openneuro_name:
        download_func = _download_via_openneuro
    elif git_url:
        download_func = _download_via_datalad
    elif osf_node:
        raise RuntimeError('OSF downloads are currently not supported.')
    elif web_url:
        download_func = _download_from_web
    else:
        raise ValueError('No download location was specified.')

    download_func(ds_name=ds_name, ds_path=ds_path)


def main(dataset):
    """Download the testing data."""
    # Save everything 'MNE_DATA' dir ... defaults to ~/mne_data
    mne_data_dir = mne.get_config(key='MNE_DATA', default=False)
    if not mne_data_dir:
        mne.set_config('MNE_DATA', str(DEFAULT_DATA_DIR))
        DEFAULT_DATA_DIR.mkdir(exist_ok=True)
        mne_data_dir = DEFAULT_DATA_DIR
    else:
        mne_data_dir = Path(mne_data_dir)

    ds_names = DATASET_OPTIONS.keys() if not dataset else (dataset,)

    for ds_name in ds_names:
        title = f'Downloading {ds_name}'
        bar = "-" * len(title)
        print(f'{title}\n{bar}')
        ds_path = mne_data_dir / ds_name
        _download(ds_name=ds_name, ds_path=ds_path)
        print()


if __name__ == '__main__':  # pragma: no cover
    # This is only used by CircleCI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='dataset', help='Name of the dataset', metavar='DATASET',
        nargs='?', default=None)
    opt = parser.parse_args()
    dataset = opt.dataset if opt.dataset != '' else None
    main(dataset)
