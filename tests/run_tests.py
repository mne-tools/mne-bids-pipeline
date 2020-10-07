"""Download test data and run a test suite."""
import sys
import os
import os.path as op
import argparse
import importlib

# Add the pipelines dir to the PATH
pipeline_dir = os.path.abspath(op.join(op.dirname(__file__), '..'))
sys.path.append(pipeline_dir)


def fetch(dataset=None):
    """Fetch the data."""
    from download_test_data import main
    main(dataset)


def sensor():
    """Run sensor pipeline."""
    mod = importlib.import_module('01-import_and_maxfilter')
    mod.main()
    mod = importlib.import_module('02-frequency_filter')
    mod.main()
    mod = importlib.import_module('03-make_epochs')
    mod.main()
    mod = importlib.import_module('04a-run_ica')
    mod.main()
    mod = importlib.import_module('04b-run_ssp')
    mod.main()
    mod = importlib.import_module('05a-apply_ica')
    mod.main()
    mod = importlib.import_module('05b-apply_ssp')
    mod.main()
    mod = importlib.import_module('06-make_evoked')
    mod.main()
    mod = importlib.import_module('07-group_average_sensors')
    mod.main()
    mod = importlib.import_module('08-sliding_estimator')
    mod.main()
    mod = importlib.import_module('09-time_frequency')
    mod.main()


def source():
    """Run source pipeline."""
    mod = importlib.import_module('10-make_forward')
    mod.main()
    mod = importlib.import_module('11-make_cov')
    mod.main()
    mod = importlib.import_module('12-make_inverse')
    mod.main()
    mod = importlib.import_module('13-group_average_source')
    mod.main()


def report():
    """Run report pipeline."""
    mod = importlib.import_module('99-make_reports')
    mod.main()


# Where to download the data to
DATA_DIR = op.join(op.expanduser('~'), 'mne_data')

TEST_SUITE = {
    'ds000246': ('config_ds000246', sensor, report),
    'ds000248': ('config_ds000248', sensor, source, report),
    'ds000248_ica': ('config_ds000248_ica', sensor, report),
    'ds001810': ('config_ds001810', sensor, report),
    'eeg_matchingpennies': ('config_eeg_matchingpennies', sensor, report),
    'ds003104': ('config_ds003104', sensor, source, report),
}


def run_tests(test_suite):
    """Run a suite of tests.

    Parameters
    ----------
    test_suite : dict
        Each key in the dict is a dataset to be tested. The associated value is
        a tuple with the first element the dataset config, and all remaining
        elements function handles to be called.

    Notes
    -----
    For every entry in the dict, the function `fetch` is called.

    """
    for dataset, test_tuple in test_suite.items():
        # export the environment variables
        os.environ['DATASET'] = dataset
        os.environ['BIDS_ROOT'] = op.join(DATA_DIR, dataset)

        config_name = test_tuple[0]
        config_path = os.path.join(pipeline_dir, 'tests', 'configs',
                                   config_name + '.py')
        os.environ['MNE_BIDS_STUDY_CONFIG'] = config_path
        del config_name, config_path

        # Fetch the data.
        fetch(dataset)

        # run the pipelines
        for pipeline in test_tuple[1::]:
            pipeline()


if __name__ == '__main__':
    # Check if we have a DATASET env var, else: inquire for one
    dataset = os.environ.get('DATASET')
    if dataset is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset', help=('dataset to test. A key in the '
                                             'TEST_SUITE dictionary. or ALL, '
                                             'to test all datasets.'))
        args = parser.parse_args()
        dataset = args.dataset

    # Triage the dataset and raise informative error if it does not exist
    if dataset == 'ALL':
        test_suite = TEST_SUITE
    else:
        test_suite = {dataset: TEST_SUITE.get(dataset, 'n/a')}

    if 'n/a' in test_suite.values():
        if os.environ.get('DATASET') is None:
            parser.print_help()
        print('\n\n')
        raise KeyError('"{}" is not a valid dataset key in the TEST_SUITE '
                       'dictionary in the run_tests.py module. Use one of {}.'
                       .format(args.dataset, ', '.join(TEST_SUITE.keys())))

    # Run the tests
    print('Running the following tests: {}'
          .format(', '.join(test_suite.keys())))

    run_tests(test_suite)
