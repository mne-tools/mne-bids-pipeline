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


class TestOptionsT(TypedDict):
    dataset: str
    config: str
    steps: Collection[str]
    task: Optional[str]
    env: Dict[str, str]


TEST_SUITE: Dict[str, TestOptionsT] = {
    'ds003392': {
        'dataset': 'ds003392',
        'config': 'config_ds003392.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'ds000246': {
        'dataset': 'ds000246',
        'config': 'config_ds000246.py',
        'steps': ('preprocessing',
                  'preprocessing/make_epochs',  # Test the group/step syntax
                  'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'ds000247': {
        'dataset': 'ds000247',
        'config': 'config_ds000247.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'rest',
        'env': {}
    },
    'ds000248': {
        'dataset': 'ds000248',
        'config': 'config_ds000248.py',
        'steps': ('preprocessing', 'sensor', 'source', 'report'),
        'task': None,
        'env': {}
    },
    'ds000248_ica': {
        'dataset': 'ds000248',
        'config': 'config_ds000248_ica.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'ds000248_T1_BEM': {
        'dataset': 'ds000248',
        'config': 'config_ds000248_T1_BEM.py',
        'steps': ('source/make_bem_surfaces',),
        'task': None,
        'env': {}
    },
    'ds000248_FLASH_BEM': {
        'dataset': 'ds000248',
        'config': 'config_ds000248_FLASH_BEM.py',
        'steps': ('source/make_bem_surfaces',),
        'task': None,
        'env': {}
    },
    'ds000248_coreg_surfaces': {
        'dataset': 'ds000248',
        'config': 'config_ds000248_coreg_surfaces.py',
        'steps': ('freesurfer/coreg_surfaces',),
        'task': None,
        'env': {}
    },
    'ds000248_no_mri': {
        'dataset': 'ds000248',
        'config': 'config_ds000248_no_mri.py',
        'steps': ('preprocessing', 'sensor', 'source', 'report'),
        'task': None,
        'env': {}
    },
    'ds001810': {
        'dataset': 'ds001810',
        'config': 'config_ds001810.py',
        'steps': ('preprocessing', 'preprocessing', 'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'eeg_matchingpennies': {
        'dataset': 'eeg_matchingpennies',
        'config': 'config_eeg_matchingpennies.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'ds003104': {
        'dataset': 'ds003104',
        'config': 'config_ds003104.py',
        'steps': ('preprocessing', 'sensor',  'source', 'report'),
        'task': None,
        'env': {}
    },
    'ds000117': {
        'dataset': 'ds000117',
        'config': 'config_ds000117.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': None,
        'env': {}
    },
    'ERP_CORE_N400': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'N400',
        'env': {}
    },
    'ERP_CORE_ERN': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'ERN',
        'env': {}
    },
    'ERP_CORE_LRP': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'LRP',
        'env': {}
    },
    'ERP_CORE_MMN': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'MMN',
        'env': {}
    },
    'ERP_CORE_N2pc': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'N2pc',
        'env': {}
    },
    'ERP_CORE_N170': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'N170',
        'env': {}
    },
    'ERP_CORE_P3': {
        'dataset': 'ERP_CORE',
        'config': 'config_ERP_CORE.py',
        'steps': ('preprocessing', 'sensor', 'report'),
        'task': 'P3',
        'env': {}
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
        if test_options['env']:
            os.environ.update(test_options['env'])

        config_path = (study_template_dir / 'tests' / 'configs' /
                       test_options['config'])

        # Fetch the data.
        if download:
            fetch(test_options['dataset'])

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
            n_jobs = '2'
        else:
            n_jobs = None

        # Run the tests.
        steps = test_options['steps']
        task = test_options['task']

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
        runpy.run_path(run_script, run_name='__main__')
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
