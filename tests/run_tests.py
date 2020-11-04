"""Download test data and run a test suite."""
import sys
import os
from pathlib import Path
import argparse
import runpy

# Add the pipelines dir to the PATH
study_template_dir = Path(__file__).absolute().parents[1]
sys.path.append(str(study_template_dir))


def fetch(dataset=None):
    """Fetch the data."""
    from download_test_data import main
    main(dataset)


# Where to download the data to
DATA_DIR = Path('~/mne_data').expanduser()

TEST_SUITE = {
    'ds000246': ('config_ds000246', 'sensor', 'report'),
    'ds000248': ('config_ds000248', 'sensor', 'source', 'report'),
    'ds000248_ica': ('config_ds000248_ica', 'sensor', 'report'),
    'ds001810': ('config_ds001810', 'sensor', 'report'),
    'eeg_matchingpennies': ('config_eeg_matchingpennies', 'sensor', 'report'),
    'ds003104': ('config_ds003104', 'sensor', 'source', 'report'),
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
        os.environ['BIDS_ROOT'] = str(DATA_DIR / dataset)

        config_name = test_tuple[0]
        config_path = (study_template_dir / 'tests' / 'configs' /
                       (config_name + '.py'))
        del config_name

        # Fetch the data.
        fetch(dataset)

        # Run the tests.
        steps = test_tuple[1:]
        run_script = study_template_dir / 'run.py'
        # We need to adjust sys.argv so we can pass "command line arguments"
        # to run.py when executed via runpy.
        argv_orig = sys.argv.copy()
        for step in steps:
            sys.argv = [sys.argv[0],
                        f'{step}',
                        f'--config={config_path}']
            # We have to use run_path because run_module doesn't allow
            # relative imports.
            runpy.run_path(run_script, run_name='__main__')
        sys.argv = argv_orig


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
