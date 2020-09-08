"""Download test data."""
import os
import os.path as op

import datalad.api as dl
import mne
from mne.commands.utils import get_optparser

DEFAULT_DATA_DIR = op.join(op.expanduser('~'), 'mne_data')


def _provide_testing_data(dataset=None):
    """Return dict of dataset, and the corresponding URLs."""
    urls_dict = {
        'eeg_matchingpennies': (
            'https://github.com/sappelhoff/eeg_matchingpennies'),
        # Anonymized "somato" dataset.
        'ds003104': 'https://github.com/OpenNeuroDatasets/ds003104',
        # Re-anonymized "ds000246" dataset.
        'ds003129': 'https://github.com/OpenNeuroDatasets/ds003129',
        'ds000248': 'https://github.com/OpenNeuroDatasets/ds000248',
        'ds000117': 'https://github.com/OpenNeuroDatasets/ds000117',
        'ds001810': 'https://github.com/OpenNeuroDatasets/ds001810',
        'ds001971': 'https://github.com/OpenNeuroDatasets/ds001971',
    }
    if dataset is None:
        return urls_dict
    else:
        return {dataset: urls_dict[dataset]}


def _provide_get_dict(dataset=None):
    """Return dict of dataset, and which data to get from it."""
    get_dict = {
        'eeg_matchingpennies': ['sub-05'],
        'ds003104': ['sub-01',
                     'derivatives/freesurfer/subjects'],
        'ds003129': ['sub-0001/meg/sub-0001_task-AEF_run-01_meg.fif',
                     'sub-0001/meg/sub-0001_task-AEF_run-01_meg.json',
                     'sub-0001/meg/sub-0001_task-AEF_run-01_channels.tsv'],
        'ds000248': ['sub-01', 'sub-emptyroom',
                     'derivatives/freesurfer/subjects'],
        'ds000117': ['sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_coordsystem.json',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos'],
        'ds001810': ['sub-01/ses-anodalpre'],
        'ds001971': ['sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_events.tsv',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.set',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.fdt',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.json',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_electrodes.tsv',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_coordsystem.json',  # noqa: E501
                     'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_channels.tsv']  # noqa: E501
    }
    if dataset is None:
        return get_dict
    else:
        return {dataset: get_dict[dataset]}


def main(dataset):
    """Download the testing data."""
    # Save everything 'MNE_DATA' dir ... defaults to ~/mne_data
    data_dir = mne.get_config(key='MNE_DATA', default=False)
    if not data_dir:
        mne.set_config('MNE_DATA', DEFAULT_DATA_DIR)
        os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
        data_dir = DEFAULT_DATA_DIR

    urls_dict = _provide_testing_data(dataset)
    get_dict = _provide_get_dict(dataset)

    for dsname, url in urls_dict.items():
        print('\n----------------------')
        dspath = op.join(data_dir, dsname)
        # install the dataset
        print('datalad installing "{}"'.format(dsname))
        dataset = dl.install(path=dspath, source=url)

        # XXX: git-annex bug: https://github.com/datalad/datalad/issues/3583
        # if datalad fails, use "get" twice, or set `n_jobs=1`
        if dsname == 'ds003104':
            n_jobs = 16
        else:
            n_jobs = 1

        # get the first subject
        for to_get in get_dict[dsname]:
            print('datalad get data "{}" for "{}"'.format(to_get, dsname))
            dataset.get(to_get, jobs=n_jobs)


if __name__ == '__main__':
    parser = get_optparser(__file__, usage="usage: %prog -dataset DATASET")
    parser.add_option('-d', '--dataset', dest='dataset',
                      help='Name of the dataset', metavar='INPUT',
                      default=None)
    opt, args = parser.parse_args()
    dataset = opt.dataset if opt.dataset != '' else None

    main(dataset)
