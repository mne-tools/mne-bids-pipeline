"""Download test data."""
import os.path as op

import datalad.api as dl


def _provide_testing_data():
    """Return URLs to testing data."""
    urls = [
        'https://github.com/OpenNeuroDatasets/ds001810',
        'https://github.com/OpenNeuroDatasets/ds001971',
    ]
    return urls


# Download the teting data
if __name__ == '__main__':
    # Save everything in ~/data
    data_dir = op.join(op.expanduser('~'), 'data')
    urls = _provide_testing_data

    for url in urls:
        # install the dataset
        dataset = dl.install(path=url, source=data_dir)

        # get the first subject
        dataset.get('sub-01')
