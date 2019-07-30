"""Download test data."""
import os
import os.path as op

import datalad.api as dl


def _provide_testing_data():
    """Return dict of dataset, and the corresponding URLs."""
    urls_dict = {
        'ds000117': 'https://github.com/OpenNeuroDatasets/ds000117',
        'ds001810': 'https://github.com/OpenNeuroDatasets/ds001810',
        'ds001971': 'https://github.com/OpenNeuroDatasets/ds001971',
    }
    return urls_dict


def _provide_get_dict():
    """Return dict of dataset, and which data to get from it."""
    get_dict = {
        'ds000117': ['sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_coordsystem.json',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif',  # noqa: E501
                     'sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos'
                     ],
        'ds001810': ['sub-01/ses-anodalpre'],
        'ds001971': ['sub-001'],
    }
    return get_dict


# Download the testing data
if __name__ == '__main__':
    # Save everything in ~/data
    data_dir = op.join(op.expanduser('~'), 'data')
    if not op.exists(data_dir):
        os.makedirs(data_dir)

    urls_dict = _provide_testing_data()
    get_dict = _provide_get_dict()

    for dsname, url in urls_dict.items():
        print('\n----------------------')
        dspath = op.join(data_dir, dsname)

        # install the dataset
        print('datalad installing "{}"'.format(dsname))
        dataset = dl.install(path=dspath, source=url)

        # get the first subject
        for to_get in get_dict[dsname]:
            print('datalad get data "{}" for "{}"'.format(to_get, dsname))
            dataset.get(to_get)
