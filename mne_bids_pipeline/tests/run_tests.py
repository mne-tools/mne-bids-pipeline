"""Download test data and run a test suite."""
import difflib
import subprocess
import sys
import shutil
import os
from pathlib import Path
from typing import Collection, Dict, Optional
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np

BIDS_PIPELINE_DIR = Path(__file__).absolute().parents[1]


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
    'ds001971': {},
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


def run_tests(test_suite, *, download):
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
        config_path = BIDS_PIPELINE_DIR / 'tests' / 'configs' / config

        # Fetch the data.
        dataset_name = test_options.get('dataset', dataset.split('_')[0])
        if download:
            fetch(dataset_name)

        # XXX Workaround for buggy date in ds000247. Remove this and the
        # XXX file referenced here once fixed!!!
        fix_path = Path(__file__).parent
        if dataset == 'ds000247':
            shutil.copy(
                src=fix_path / 'ds000247_scans.tsv',
                dst=Path('~/mne_data/ds000247/sub-0002/ses-01/'
                         'sub-0002_ses-01_scans.tsv').expanduser()
            )
        # XXX Workaround for buggy participant_id in ds001971
        elif dataset == 'ds001971':
            shutil.copy(
                src=fix_path / 'ds001971_participants.tsv',
                dst=Path('~/mne_data/ds001971/participants.tsv').expanduser()
            )

        # Run the tests.
        steps = test_options.get(
            'steps', ('preprocessing', 'sensor', 'report'))
        task = test_options.get('task', None)
        command = [
            'mne_bids_pipeline',
            f'--config={config_path}',
            f'--steps={",".join(steps)}',
            f'--task={task}' if task else '']
        command.extend(sys.argv[1:])
        command = [x for x in command if x != '']  # Eliminate "empty" items
        subprocess.check_call(command)


if __name__ == '__main__':
    # Don't argparse, just take the first arg, --download (if present), and
    # pass all the rest
    try:
        download = sys.argv.index('--download')
    except ValueError:
        download = False
    else:
        sys.argv.pop(download)
        download = True
    which = np.where([not x.startswith('-') for x in sys.argv[1:]])[0]
    dataset = [sys.argv[w + 1] for w in which]
    if len(dataset) != 1:
        raise RuntimeError(
            'run_tests.py requires exactly one argument, the dataset name, '
            f'got {dataset}')
    dataset = sys.argv.pop(which[0] + 1)
    if dataset != 'ALL' and dataset not in TEST_SUITE:
        print('\n')
        extra = ''
        matches = difflib.get_close_matches(dataset, TEST_SUITE.keys())
        if matches:
            extra = f'\n\nDid you mean one of {matches}?'
        print(
            f'\n\n{repr(dataset)}" is not a valid dataset key in the '
            f'TEST_SUITE dictionary in the run_tests.py module.{extra}\n\n'
            f'Valid options: {", ".join(sorted(TEST_SUITE.keys()))}.\n'
        )
        raise KeyError(f'{repr(dataset)} is not a valid dataset key.')

    if dataset == 'ALL':
        test_suite = TEST_SUITE
    else:
        test_suite = {dataset: TEST_SUITE[dataset]}

    # Run the tests
    extra = ' after downloading data' if download else ''
    print(f'📝 Running the following tests{extra}: '
          f'{", ".join(test_suite.keys())}')

    run_tests(test_suite, download=download)
