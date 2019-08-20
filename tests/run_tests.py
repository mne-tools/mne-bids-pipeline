"""Download test data and run a test suite."""
import os
import os.path as op
import subprocess
import argparse

# Where to download the data to
DATA_DIR = op.join(op.expanduser('~'), 'mne_data')

TEST_SUITE = {
    'ds000246': ('config_ds000246', 'fetch', 'sensor'),
    'ds000248': ('config_ds000248', 'fetch', 'sensor'),
    'ds001810': ('config_ds001810', 'fetch', 'sensor'),
    'eeg_matchingpennies': ('config_matchingpennies', 'fetch', 'sensor'),
    'somato': ('config_somato', 'fetch', 'sensor', 'source'),
}


def run_tests(test_suite):
    """Run a suite of tests.

    Parameters
    ----------
    test_suite : dict
        Each key in the dict is a dataset to be tested. The associated value is
        a tuple with the first element the dataset config, and all remaining
        elements the commands to run with GNU Make.

    """
    for dataset, test_tuple in test_suite.items():
        # export the environment variables
        os.environ['DATASET'] = dataset
        os.environ['BIDS_ROOT'] = op.join(DATA_DIR, dataset)
        os.environ['MNE_BIDS_STUDY_CONFIG'] = test_tuple[0]

        # run GNU Make
        for make_cmd in test_tuple[1::]:
            subprocess.run(['make', make_cmd], check=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help=('dataset to test. A key in the '
                                         'TEST_SUITE dictionary. or ALL, '
                                         'to test all datasets.'))
    args = parser.parse_args()

    if args.dataset == 'ALL':
        test_suite = TEST_SUITE
    else:
        test_suite = {args.dataset: TEST_SUITE.get(args.dataset, 'n/a')}

    if 'n/a' in test_suite.values():
        parser.print_help()
        print('\n\n')
        raise KeyError('"{}" is not a valid dataset key in the TEST_SUITE '
                       'dictionary in the run_tests.py module.'
                       .format(args.dataset))
    else:
        # Run the tests
        print('Running the following tests:\n')
        for dataset, test_tuple in test_suite.items():
            print(dataset, test_tuple)
        run_tests(test_suite)
