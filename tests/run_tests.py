"""Download test data and run a test suite."""
import sys
import shutil
import os
from pathlib import Path
import argparse
import runpy
from typing import Collection, Dict, Optional
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# Add the pipelines dir to the PATH
study_template_dir = Path(__file__).absolute().parents[1]
sys.path.append(str(study_template_dir))


def fetch(dataset=None):
    """Fetch the data."""
    from download_test_data import main
    main(dataset)


# Where to download the data to
DATA_DIR = Path('~/mne_data').expanduser()


# Once PEP655 lands in 3.11 we can use NotRequired instead of total=False
class TestOptionsT(TypedDict, total=False):
    dataset: str
    config: str
    steps: Collection[str]
    task: Optional[str]
    env: Dict[str, str]


# If not supplied below, the defaults are:
# key: {
#     'dataset': key.split('_')[0],
#     'config': f'config_{key}.py',
#     'steps': ('preprocessing', 'sensor', 'report'),
#     'env': {},
#     'task': None,
# }
#
TEST_SUITE: Dict[str, TestOptionsT] = {
    'ds003392': {},
    'ds004229': {},
    #'ds001971': {},
    'ds004107': {},
    'ds000117': {},
    'ds003775': {},
    'eeg_matchingpennies': {
        'dataset': 'eeg_matchingpennies',
    },
    'ds000246': {
        'steps': ('preprocessing',
                  'preprocessing/make_epochs',  # Test the group/step syntax
                  'sensor', 'report'),
    },
    'ds000247': {
        'task': 'rest',
    },
    'ds000248': {
        'steps': ('preprocessing', 'sensor', 'source', 'report'),
    },
    'ds000248_ica': {},
    'ds000248_T1_BEM': {
        'steps': ('source/make_bem_surfaces',),
    },
    'ds000248_FLASH_BEM': {
        'steps': ('source/make_bem_surfaces',),
    },
    'ds000248_coreg_surfaces': {
        'steps': ('freesurfer/coreg_surfaces',),
    },
    'ds000248_no_mri': {
        'steps': ('preprocessing', 'sensor', 'source', 'report'),
    },
    'ds001810': {
        'steps': ('preprocessing', 'preprocessing', 'sensor', 'report'),
    },
    'ds003104': {
        'steps': ('preprocessing', 'sensor',  'source', 'report'),
    },
    'ERP_CORE_N400': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'N400',
    },
    'ERP_CORE_ERN': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'ERN',
    },
    'ERP_CORE_LRP': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'LRP',
    },
    'ERP_CORE_MMN': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'MMN',
    },
    'ERP_CORE_N2pc': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'N2pc',
    },
    'ERP_CORE_N170': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'N170',
    },
    'ERP_CORE_P3': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'task': 'P3',
    }
}


def run_tests(test_suite, download):
    """Run a suite of tests.

    Parameters
    ----------
    test_suite : dict
        Each key in the dict is a dataset to be tested. The associated value is
        a tuple with the first element the dataset config, and all remaining
        elements function handles to be called.
    download : bool
        Whether to (re-)download the test dataset.

    Notes
    -----
    For every entry in the dict, the function `fetch` is called.

    """
    for dataset, test_options in test_suite.items():
        # export the environment variables
        os.environ['DATASET'] = dataset
        if 'env' in test_options:
            os.environ.update(test_options['env'])

        config = test_options.get('config', f'config_{dataset}.py')
        config_path = study_template_dir / 'tests' / 'configs' / config

        # Fetch the data.
        dataset_name = test_options.get('dataset', dataset.split('_')[0])
        if download:
            fetch(dataset_name)

        # XXX Workaround for buggy date in ds000247. Remove this and the
        # XXX file referenced here once fixed!!!
        if dataset == 'ds000247':
            shutil.copy(
                src=Path(__file__).parent / 'ds000247_scans.tsv',
                dst=Path('~/mne_data/ds000247/sub-0002/ses-01/'
                         'sub-0002_ses-01_scans.tsv').expanduser()
            )

        # Test the `--n_jobs` parameter
        if dataset == 'ds000117':
            n_jobs = '1'
        else:
            n_jobs = None

        # Run the tests.
        steps = test_options.get(
            'steps', ('preprocessing', 'sensor', 'report'))
        task = test_options.get('task', None)

        run_script = study_template_dir / 'run.py'
        # We need to adjust sys.argv so we can pass "command line arguments"
        # to run.py when executed via runpy.
        argv_orig = sys.argv.copy()
        sys.argv = [
            sys.argv[0],
            f'--steps={",".join(steps)}',
            f'--config={config_path}',
            f'--task={task}' if task else '',
            f'--n_jobs={n_jobs}' if n_jobs else '',
            f'--interactive=0'
        ]
        # Eliminate "empty" items
        sys.argv = [arg for arg in sys.argv if arg != '']
        # We have to use run_path because run_module doesn't allow
        # relative imports.
        runpy.run_path(str(run_script), run_name='__main__')
        sys.argv = argv_orig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to test. A key in the '
                                        'TEST_SUITE dictionary, or ALL, '
                                        'to test all datasets.')
    parser.add_argument('--download', choices=['0', '1'],
                        help='Whether to (re-)download the dataset.')
    args = parser.parse_args()
    dataset = args.dataset
    download = args.download
    download = True if download is None else bool(int(download))
    # Triage the dataset and raise informative error if it does not exist
    if dataset == 'ALL':
        test_suite = TEST_SUITE
    else:
        test_suite = {dataset: TEST_SUITE.get(dataset, 'n/a')}

    if 'n/a' in test_suite.values():
        if os.environ.get('DATASET') is None:
            parser.print_help()
        print('\n\n')
        raise KeyError(
            f'"{args.dataset}" is not a valid dataset key in the TEST_SUITE '
            f'dictionary in the run_tests.py module. Use one of: '
            f'{", ".join(TEST_SUITE.keys())}.'
        )

    # Run the tests
    print(f'Running the following tests: {", ".join(test_suite.keys())}')

    run_tests(test_suite, download=download)
