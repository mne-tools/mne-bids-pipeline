"""Download test data and run a test suite."""

import os
import re
import shutil
import sys
from collections.abc import Collection, Generator
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypedDict

import pytest
from h5io import read_hdf5
from mne_bids import BIDSPath, get_bids_path_from_fname

from mne_bids_pipeline._config_import import _import_config
from mne_bids_pipeline._download import main as download_main
from mne_bids_pipeline._main import main

BIDS_PIPELINE_DIR = Path(__file__).absolute().parents[1]


# Where to download the data to
DATA_DIR = Path("~/mne_data").expanduser()


# Once PEP655 lands in 3.11 we can use NotRequired instead of total=False
# Effective defaults are listed in comments
class _TestOptionsT(TypedDict, total=False):
    dataset: str  # key.split("_")[0]
    config: str  # f"config_{key}.py"
    steps: Collection[str]  # ("preprocessing", "sensor")
    task: str | None  # None
    env: dict[str, str]  # {}
    requires: Collection[str]  # ()
    extra_config: str  # ""


TEST_SUITE: dict[str, _TestOptionsT] = {
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
    "ds000248_ica": {
        "extra_config": """
_raw_split_size = "60MB"
_epochs_split_size = "30MB"
_n_jobs = {}
"""
    },
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
    "MNE-phantom-KIT-data": {
        "config": "config_MNE_phantom_KIT_data.py",
    },
    "MNE-funloc-data": {
        "config": "config_MNE_funloc_data.py",
        "steps": ["init", "preprocessing", "sensor", "source"],
    },
}


@pytest.fixture()
def dataset_test(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Provide a defined context for our dataset tests."""
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
def test_run(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
    dataset_test: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    elif dataset == "ds003775":
        shutil.copy(
            src=fix_path / "sub-010_ses-t1_scans.tsv",
            dst=DATA_DIR
            / "ds003775"
            / "sub-010"
            / "ses-t1"
            / "sub-010_ses-t1_scans.tsv",
        )

    # Run the tests.
    steps = test_options.get("steps", ("preprocessing", "sensor"))
    task = test_options.get("task", None)
    command = ["mne_bids_pipeline", str(config_path), f"--steps={','.join(steps)}"]
    if task:
        command.append(f"--task={task}")
    if "--pdb" in sys.argv:
        command.append("--n_jobs=1")
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING", "true")
    monkeypatch.setattr(sys, "argv", command)
    with capsys.disabled():
        print()
        main()


@pytest.mark.parametrize("allow_missing_sessions", (False, True))
def test_missing_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    allow_missing_sessions: bool,
) -> None:
    """Test the `allow_missing_sessions` config variable."""
    dataset = "fake"
    bids_root = tmp_path / dataset
    files = (
        "dataset_description.json",
        *(f"participants.{x}" for x in ("json", "tsv")),
        *(f"sub-1/sub-1_sessions.{x}" for x in ("json", "tsv")),
        *(
            f"sub-1/ses-a/eeg/sub-1_ses-a_task-foo_{x}.tsv"
            for x in ("channels", "events")
        ),
        *(
            f"sub-1/ses-a/eeg/sub-1_ses-a_task-foo_eeg.{x}"
            for x in ("eeg", "json", "vhdr", "vmrk")
        ),
    )
    for _file in files:
        path = bids_root / _file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # fake a config file (can't use static file because `bids_root` is in `tmp_path`)
    config = f"""
bids_root = "{bids_root}"
deriv_root = "{tmp_path / "derivatives" / "mne-bids-pipeline" / dataset}"
interactive = False
subjects = ["1"]
sessions = ["a", "b"]
ch_types = ["eeg"]
conditions = ["zzz"]
allow_missing_sessions = {allow_missing_sessions}
"""
    config_path = tmp_path / "fake_config_missing_session.py"
    with open(config_path, "w") as fid:
        fid.write(config)
    # set up the context handler
    context = (
        nullcontext()
        if allow_missing_sessions
        else pytest.raises(RuntimeError, match=r"Subject 1 is missing session \['b'\]")
    )
    # run
    command = [
        "mne_bids_pipeline",
        str(config_path),
        "--steps=init/_01_init_derivatives_dir",
    ]
    if "--pdb" in sys.argv:
        command.append("--n_jobs=1")
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING", "true")
    monkeypatch.setattr(sys, "argv", command)
    with capsys.disabled():
        print()
        with context:
            main()


@pytest.mark.dataset_test
def test_session_specific_mri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test of (faked) session-specific MRIs."""
    dataset = "MNE-funloc-data"
    test_options = TEST_SUITE[dataset]
    config = test_options.get("config", f"config_{dataset}.py")
    config_path = BIDS_PIPELINE_DIR / "tests" / "configs" / config
    config_obj = _import_config(config_path=config_path)
    # copy the dataset to a tmpdir, and in the destination location make it
    # seem like there's only one subj with different MRIs for different sessions
    new_bids_path = BIDSPath(root=tmp_path / dataset, subject="01", session="a")
    assert new_bids_path.root is not None
    # sub-01/* → sub-01/ses-a/* ;  sub-02/* → sub-01/ses-b/*
    for src_subj, dst_sess in (("01", "a"), ("02", "b")):
        src_dir = config_obj.bids_root / f"sub-{src_subj}"
        dst_dir = new_bids_path.root / "sub-01" / f"ses-{dst_sess}"
        for walk_root, dirs, files in src_dir.walk():
            offset = walk_root.relative_to(src_dir)
            for _dir in dirs:
                (dst_dir / offset / _dir).mkdir(parents=True)
            for _file in files:
                bp = get_bids_path_from_fname(walk_root / _file)
                bp.update(root=new_bids_path.root, subject="01", session=dst_sess)
                # rewrite scans.tsv files to have correct filenames in it
                if _file.endswith("scans.tsv"):
                    lines = [
                        line.replace(f"sub-{src_subj}", f"sub-01_ses-{dst_sess}")
                        for line in (walk_root / _file).read_text().split("\n")
                    ]
                    (dst_dir / offset / bp.basename).write_text("\n".join(lines))
                # For all other files, a simple copy suffices; rewriting
                # `raw.info["subject_info"]["his_id"]` is not necessary because MNE-BIDS
                # overwrites it with the value in `participants.tsv` anyway.
                else:
                    shutil.copyfile(
                        src=walk_root / _file, dst=dst_dir / offset / bp.basename
                    )
    # emptyroom
    src_dir = config_obj.bids_root / "sub-emptyroom"
    dst_dir = new_bids_path.root / "sub-emptyroom"
    shutil.copytree(src=src_dir, dst=dst_dir)
    # root-level files (dataset description, etc)
    src_dir = config_obj.bids_root
    dst_dir = new_bids_path.root
    files = [f for f in src_dir.iterdir() if f.is_file()]
    for _file in files:
        # in theory we should rewrite `participants.tsv` to remove the `sub-02` line,
        # but in practice it will just get ignored so we won't bother.
        shutil.copyfile(src=_file, dst=dst_dir / _file.name)
    # derivatives (freesurfer files)
    src_dir = config_obj.bids_root / "derivatives" / "freesurfer" / "subjects"
    dst_dir = new_bids_path.root / "derivatives" / "freesurfer" / "subjects"
    dst_dir.mkdir(parents=True)
    freesurfer_subject_mapping = {"sub-01": "sub-01_ses-a", "sub-02": "sub-01_ses-b"}
    for walk_root, dirs, files in src_dir.walk():
        # change "root" so that in later steps of the walk when we're inside a subject's
        # dir, the "offset" (folders between dst_dir and filename) will be correct
        new_root = walk_root
        if "sub-01" in walk_root.parts or "sub-02" in walk_root.parts:
            new_root = Path(
                *[freesurfer_subject_mapping.get(p, p) for p in new_root.parts]
            )
        offset = new_root.relative_to(src_dir)
        # the actual subject dirs need their names changed
        for _dir in dirs:
            _dir = freesurfer_subject_mapping.get(_dir, _dir)
            (dst_dir / offset / _dir).mkdir()
        # for filenames that contain the subject identifier (BEM files, morph maps),
        # we need to change the filename too, not just parent folder name
        for _file in files:
            dst_file = _file
            for subj in freesurfer_subject_mapping:
                if subj in dst_file:
                    dst_file = dst_file.replace(subj, freesurfer_subject_mapping[subj])
                    break
            shutil.copyfile(src=walk_root / _file, dst=dst_dir / offset / dst_file)
    # update config so that `subjects_dir` and `deriv_root` also point to the tempdir
    extra_config = f"""
from pathlib import Path
subjects_dir = "{new_bids_path.root / "derivatives" / "freesurfer" / "subjects"}"
deriv_root = Path("{new_bids_path.root}") / "derivatives" / "mne-bids-pipeline" / "MNE-funloc-data"
"""  # noqa E501
    extra_path = tmp_path / "extra_config.py"
    extra_path.write_text(extra_config)
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING_EXTRA_CONFIG", str(extra_path))
    # Run the tests.
    steps = test_options.get("steps", ())
    command = ["mne_bids_pipeline", str(config_path), f"--steps={','.join(steps)}"]
    # hack in the new bids_root
    command.append(f"--root-dir={new_bids_path.root}")
    if "--pdb" in sys.argv:
        command.append("--n_jobs=1")
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING", "true")
    monkeypatch.setattr(sys, "argv", command)
    with capsys.disabled():
        print()
        main()
    # check some things that are indicative of different MRIs being used in each session
    results = list()
    for sess in ("a", "b"):
        fname = (
            new_bids_path.root
            / "derivatives"
            / "mne-bids-pipeline"
            / "MNE-funloc-data"
            / "sub-01"
            / f"ses-{sess}"
            / "meg"
            / f"sub-01_ses-{sess}_task-funloc_report.h5"
        )
        report = read_hdf5(fname, title="mnepython")
        coregs = next(
            filter(lambda x: x["dom_id"] == "Sensor_alignment", report["_content"])
        )
        pattern = re.compile(
            r"Average distance from (?P<npts>\d+) digitized points to head: "
            r"(?P<dist>\d+(?:\.\d+)?) mm"
        )
        result = pattern.search(coregs["html"])
        assert result is not None
        assert float(result.group("dist")) < 3  # fit between pts and outer_skin < 3 mm
        results.append(result.groups())
    assert results[0] != results[1]  # different npts and/or different mean distance
