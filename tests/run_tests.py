"""Download test data and run a test suite."""
import os
import os.path as op
import subprocess

# Where to download the data to
DATA_DIR = op.join(op.expanduser('~'), 'mne_data')

TEST_SUITE = [
    ('ds000246', 'config_ds000246', 'fetch', 'sensor'),
    ('ds000248', 'config_ds000248', 'fetch', 'sensor'),
    ('ds001810', 'config_ds001810', 'fetch', 'sensor'),
    ('eeg_matchingpennies', 'config_matchingpennies', 'fetch', 'sensor'),
    ('somato', 'config_somato', 'fetch', 'sensor', 'source'),
]


def run_tests(test_suite):
    """Run a suite of tests.

    Parameters
    ----------
    test_suite : list of tuples
        Each tuple is a test with the first element the dataset, second element
        the dataset config, and all remaining elements the commands to run with
        GNU Make.

    """
    for test_tuple in test_suite:
        # export the environment variables
        os.environ['DATASET'] = test_tuple[0]
        os.environ['BIDS_ROOT'] = op.join(DATA_DIR, test_tuple[0])
        os.environ['MNE_BIDS_STUDY_CONFIG'] = test_tuple[1]

        # run GNU Make
        for make_cmd in test_tuple[2::]:
            subprocess.run(['make', make_cmd], check=True)


if __name__ == '__main__':
    run_tests(TEST_SUITE)
