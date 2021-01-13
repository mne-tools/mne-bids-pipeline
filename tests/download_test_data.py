"""Download test data."""
from pathlib import Path

import openneuro
import mne
from mne.commands.utils import get_optparser

from datasets import DATASET_OPTIONS

DEFAULT_DATA_DIR = Path('~/mne_data').expanduser()


def _download_via_datalad(*, ds_name, ds_path):
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


def _download_via_openneuro(*, ds_name, ds_path):
    openneuro.download(dataset=DATASET_OPTIONS[ds_name]['openneuro'],
                       target_dir=ds_path,
                       include=DATASET_OPTIONS[ds_name]['include'],
                       exclude=DATASET_OPTIONS[ds_name]['exclude'])


def _download(*, ds_name, ds_path):
    openneuro_name = DATASET_OPTIONS[ds_name]['openneuro']
    git_url = DATASET_OPTIONS[ds_name]['git']

    if not openneuro_name and not git_url:
        raise ValueError(f'Neither git URL nor OpenNeuro dataset name known, '
                         f'cannot download dataset {ds_name}')

    download_func = (_download_via_openneuro if openneuro_name
                     else _download_via_datalad)
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
        print('\n----------------------')
        ds_path = mne_data_dir / ds_name
        _download(ds_name=ds_name, ds_path=ds_path)


if __name__ == '__main__':
    parser = get_optparser(__file__, usage="usage: %prog -dataset DATASET")
    parser.add_option('-d', '--dataset', dest='dataset',
                      help='Name of the dataset', metavar='INPUT',
                      default=None)
    opt, args = parser.parse_args()
    dataset = opt.dataset if opt.dataset != '' else None

    main(dataset)
