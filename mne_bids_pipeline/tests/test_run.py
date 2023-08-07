"""Download test data and run a test suite."""
import sys
import shutil
from pathlib import Path
from typing import Collection, Dict, Optional, TypedDict
import os

import pytest

from mne_bids_pipeline._main import main
from mne_bids_pipeline._download import main as download_main

BIDS_PIPELINE_DIR = Path(__file__).absolute().parents[1]


# Where to download the data to
DATA_DIR = Path("~/mne_data").expanduser()


# Once PEP655 lands in 3.11 we can use NotRequired instead of total=False
# Effective defaults are listed in comments
class _TestOptionsT(TypedDict, total=False):
    dataset: str  # key.split("_")[0]
    config: str  # f"config_{key}.py"
    steps: Collection[str]  # ("preprocessing", "sensor")
    task: Optional[str]  # None
    env: Dict[str, str]  # {}
    requires: Collection[str]  # ()
    extra_config: str  # ""


TEST_SUITE: Dict[str, _TestOptionsT] = {
    "ds003392": {},
    "ds004229": {},
    "ds001971": {},
    "ds004107": {},
    "ds000117": {},
    "ds003775": {},
    "eeg_matchingpennies": {
        "dataset": "eeg_matchingpennies",
    },
    "ds000246": {
        "steps": (
            "preprocessing",
            "preprocessing/make_epochs",  # Test the group/step syntax
            "sensor",
        ),
    },
    "ds000247": {
        "task": "rest",
    },
    "ds000248_base": {
        "steps": ("preprocessing", "sensor", "source"),
        "requires": ("freesurfer",),
        "extra_config": """
_raw_split_size = "60MB"  # hits both task-noise and task-audiovisual
_epochs_split_size = "30MB"
# use n_jobs=1 here to ensure that we get coverage for metadata_query
_n_jobs = {"preprocessing/_05_make_epochs": 1}
""",
    },
    "ds000248_ica": {},
    "ds000248_T1_BEM": {
        "steps": ("source/make_bem_surfaces",),
        "requires": ("freesurfer",),
    },
    "ds000248_FLASH_BEM": {
        "steps": ("source/make_bem_surfaces",),
        "requires": ("freesurfer",),
    },
    "ds000248_coreg_surfaces": {
        "steps": ("freesurfer/coreg_surfaces",),
        "requires": ("freesurfer",),
    },
    "ds000248_no_mri": {
        "steps": ("preprocessing", "sensor", "source"),
    },
    "ds001810": {
        "steps": ("preprocessing", "preprocessing", "sensor"),
    },
    "ds003104": {
        "steps": ("preprocessing", "sensor", "source"),
    },
    "ERP_CORE_N400": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "N400",
    },
    "ERP_CORE_ERN": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "ERN",
        "extra_config": """
# use n_jobs = 1 with loky to ensure that the CSP steps get proper coverage
_n_jobs = {
    "sensor/_05_decoding_csp": 1,
    "sensor/_99_group_average": 1,
}
""",
    },
    "ERP_CORE_LRP": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "LRP",
    },
    "ERP_CORE_MMN": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "MMN",
    },
    "ERP_CORE_N2pc": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "N2pc",
    },
    "ERP_CORE_N170": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "N170",
    },
    "ERP_CORE_P3": {
        "dataset": "ERP_CORE",
        "config": "config_ERP_CORE.py",
        "task": "P3",
    },
}


@pytest.fixture()
def dataset_test(request):
    # There is probably a cleaner way to get this param, but this works for now
    capsys = request.getfixturevalue("capsys")
    dataset = request.getfixturevalue("dataset")
    test_options = TEST_SUITE[dataset]
    if "freesurfer" in test_options.get("requires", ()):
        if "FREESURFER_HOME" not in os.environ:
            pytest.skip("FREESURFER_HOME required but not found")
    dataset_name = test_options.get("dataset", dataset.split("_")[0])
    with capsys.disabled():
        if request.config.getoption("--download", False):  # download requested
            download_main(dataset_name)
        yield


@pytest.mark.dataset_test
@pytest.mark.parametrize("dataset", list(TEST_SUITE))
def test_run(dataset, monkeypatch, dataset_test, capsys, tmp_path):
    """Test running a dataset."""
    test_options = TEST_SUITE[dataset]
    config = test_options.get("config", f"config_{dataset}.py")
    config_path = BIDS_PIPELINE_DIR / "tests" / "configs" / config
    extra_config = TEST_SUITE[dataset].get("extra_config", "")
    if extra_config:
        extra_path = tmp_path / "extra_config.py"
        extra_path.write_text(extra_config)
        monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING_EXTRA_CONFIG", str(extra_path))

    # XXX Workaround for buggy date in ds000247. Remove this and the
    # XXX file referenced here once fixed!!!
    fix_path = Path(__file__).parent
    if dataset == "ds000247":
        dst = (
            DATA_DIR / "ds000247" / "sub-0002" / "ses-01" / "sub-0002_ses-01_scans.tsv"
        )
        shutil.copy(src=fix_path / "ds000247_scans.tsv", dst=dst)
    # XXX Workaround for buggy participant_id in ds001971
    elif dataset == "ds001971":
        shutil.copy(
            src=fix_path / "ds001971_participants.tsv",
            dst=DATA_DIR / "ds001971" / "participants.tsv",
        )

    # Run the tests.
    steps = test_options.get("steps", ("preprocessing", "sensor"))
    task = test_options.get("task", None)
    command = ["mne_bids_pipeline", str(config_path), f'--steps={",".join(steps)}']
    if task:
        command.append(f"--task={task}")
    if "--pdb" in sys.argv:
        command.append("--n_jobs=1")
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING", "true")
    monkeypatch.setattr(sys, "argv", command)
    with capsys.disabled():
        print()
        main()
