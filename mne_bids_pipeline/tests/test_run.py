"""Download test data and run a test suite."""

import contextlib
import os
import re
import shutil
import sys
import warnings
from collections.abc import Collection, Generator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TypedDict

import mne
import pandas as pd
import pytest
from h5io import read_hdf5
from mne_bids import BIDSPath, get_bids_path_from_fname

from mne_bids_pipeline._config_import import _import_config
from mne_bids_pipeline._config_utils import _get_ssrt
from mne_bids_pipeline._download import main as download_main
from mne_bids_pipeline._main import main
from mne_bids_pipeline._report import _run_sort_key
from mne_bids_pipeline.steps.freesurfer import _01_recon_all
from mne_bids_pipeline.steps.preprocessing._01a_data_quality import (
    get_config as get_config_data_quality,
)
from mne_bids_pipeline.steps.preprocessing._01a_data_quality import (
    get_input_fnames_data_quality,
)

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
    "ds003392_base": {
        # "dataset": "ds003392",
    },
    "ds003392_otp_mxw": {
        "steps": ("preprocessing",),
        # "dataset": "ds003392",
    },
    "ds003392_otp_ff": {
        "steps": ("preprocessing",),
        # "dataset": "ds003392",
    },
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


class _ReportTOCFinder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_a = False
        self.in_toc = False
        self.toc_links = list()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.in_a = True
        elif tag == "div" and ("id", "toc") in attrs:
            self.in_toc = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.in_a = False
        elif tag == "div" and self.in_toc:
            self.in_toc = False

    def handle_data(self, data: str) -> None:
        if self.in_a and self.in_toc:
            self.toc_links.append(data)


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

    warning_ctx = contextlib.nullcontext
    fix_path = Path(__file__).parent
    if dataset == "ds000247":
        # XXX Workaround for buggy date in ds000247. Remove this and the
        # XXX file referenced here once fixed!!!
        dst = (
            DATA_DIR / "ds000247" / "sub-0002" / "ses-01" / "sub-0002_ses-01_scans.tsv"
        )
        shutil.copy(src=fix_path / "ds000247_scans.tsv", dst=dst)
    elif dataset == "ds001971":
        # XXX Workaround for buggy participant_id in ds001971
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
    elif dataset == "ds004229":

        @contextlib.contextmanager
        def warning_ctx():
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings(
                    "ignore", ".*SVD did not converge.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", ".*cannot determine the transf.*", category=RuntimeWarning
                )
                yield

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
    with capsys.disabled(), warning_ctx():
        print()
        main()

    # post-run checks for correctness
    config_data = config_path.read_text("utf-8")

    # sub-average evoked present in report
    has_evoked_conditions = (
        re.search(r"^\s*conditions =", config_data, flags=re.MULTILINE) is not None
    )
    if "sensor" in steps and has_evoked_conditions:
        assert dataset not in ("ds000247", "ds000375")
        ds_path = test_options.get("dataset", dataset)
        avg_subj_path = (
            DATA_DIR / "derivatives" / "mne-bids-pipeline" / ds_path / "sub-average"
        )
        assert avg_subj_path.is_dir()
        if ds_path == "ERP_CORE":
            avg_subj_path = avg_subj_path / f"ses-{test_options['task']}"
            assert avg_subj_path.is_dir()
        report_html_paths = list(avg_subj_path.rglob("sub-average*_report.html"))
        assert len(report_html_paths)
        parser = _ReportTOCFinder()
        parser.feed(report_html_paths[0].read_text("utf-8"))
        msg = "\n".join(["Not found in TOC titles:"] + parser.toc_links)
        assert any("Average (sensor)" in name for name in parser.toc_links), msg
    else:
        # Just spot check a few that we know have "conditions" to make sure our
        # conditional is good
        assert dataset not in ("ds000248", "ds004229", "ERP_CORE_P3")

    # every report gets a flow diagram, replaced in place across report saves
    deriv_path = (
        DATA_DIR
        / "derivatives"
        / "mne-bids-pipeline"
        / test_options.get("dataset", dataset)
    )
    report_html_paths = sorted(deriv_path.rglob("sub-*_report.html"))
    if "preprocessing" not in steps:
        # FreeSurfer/BEM-only runs may write no report, or an edge-less recording
        # (e.g. coreg_surfaces consuming its own cached seghead) and no diagram
        for report_html_path in report_html_paths[:1]:
            content = report_html_path.read_text("utf-8")
            assert content.count('<svg id="mbp-flow-svg"') <= 1
    else:
        assert len(report_html_paths)
        content = report_html_paths[0].read_text("utf-8")
        assert content.count('<svg id="mbp-flow-svg"') == 1
        assert content.count('aria-label="Pipeline flow diagram"') == 1
        assert "mbp-flow-cat-" in content  # step-category CSS included
        # tooltip paths are shortened against the recorded roots, which the
        # source node's tooltip defines
        assert "&lt;deriv_root&gt; = " in content
        assert "&lt;bids_root&gt;" in content
        # run sections appear in run order regardless of parallel completion
        # order (gh-845)
        report = mne.open_report(report_html_paths[0].with_suffix(".h5"))
        titles, all_tags, _ = report.get_contents()
        by_prefix: dict[str, list[str]] = dict()
        for title, tags in zip(titles, all_tags):
            for tag in tags:
                if tag.startswith("run-"):
                    run = tag.removeprefix("run-")
                    prefix = title.replace(f"run-{run}", "run-")
                    by_prefix.setdefault(prefix, []).append(run)
                    break
        for prefix, runs in by_prefix.items():
            keys = [_run_sort_key(run) for run in runs]
            assert keys == sorted(keys), (prefix, runs)


def _make_fake_bids_dataset(
    bids_root: Path,
    *,
    subjects: list[str],
    task: str,
    sessions: tuple[str | None, ...] = (None,),
    runs: tuple[str | None, ...] = (None,),
    suffixes: tuple[str, ...] = (
        "channels.tsv",
        "events.tsv",
        "eeg.vhdr",
        "eeg.vmrk",
        "eeg.eeg",
        "eeg.json",
    ),
) -> None:
    """Create a minimal fake BIDS EEG dataset with empty (invalid) data files.

    The files exist so mne_bids can find them, but their content is empty, so
    actually reading them fails "for real" if a test goes that far. This matters
    for on_error tests: the failure needs to reproduce even inside separate
    (loky) worker processes, which a monkeypatched function would not.
    """
    for subject in subjects:
        sub = f"sub-{subject}"
        for session in sessions:
            ses_entity = f"_ses-{session}" if session is not None else ""
            eeg_dir = (
                f"{sub}/ses-{session}/eeg" if session is not None else f"{sub}/eeg"
            )
            for run in runs:
                run_entity = f"_run-{run}" if run is not None else ""
                stem = f"{sub}{ses_entity}_task-{task}{run_entity}"
                for suffix in suffixes:
                    file_path = bids_root / eeg_dir / f"{stem}_{suffix}"
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.touch()
    for _file in ("dataset_description.json", "participants.tsv", "participants.json"):
        (bids_root / _file).touch()


def _write_pipeline_config(
    config_path: Path,
    *,
    bids_root: Path,
    deriv_root: Path,
    ch_types: list[str] | None = None,
    conditions: list[str] | None = None,
    **extra: Any,
) -> None:
    """Write a minimal pipeline config file from keyword arguments.

    `ch_types`/`conditions` default to the minimal values used across our
    fake-dataset tests. Any other pipeline config option can be passed via
    `extra`, e.g. `subjects=[...]`, `task=...`, `on_error=...`,
    `allow_missing_sessions=...`.
    """
    config: dict[str, Any] = dict(
        bids_root=bids_root,
        deriv_root=deriv_root,
        ch_types=["eeg"] if ch_types is None else ch_types,
        conditions=["zzz"] if conditions is None else conditions,
        **extra,
    )
    lines = []
    for key, val in config.items():
        if isinstance(val, Path):
            val = str(val)
        lines.append(f"{key} = {val!r}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    # Only session "a" gets data on disk; session "b" is deliberately missing.
    _make_fake_bids_dataset(bids_root, subjects=["1"], task="foo", sessions=("a",))
    for suffix in ("json", "tsv"):
        (bids_root / "sub-1" / f"sub-1_sessions.{suffix}").touch()
    # fake a config file (can't use static file because `bids_root` is in `tmp_path`)
    config_path = tmp_path / "fake_config_missing_session.py"
    _write_pipeline_config(
        config_path,
        bids_root=bids_root,
        deriv_root=tmp_path / "derivatives" / "mne-bids-pipeline" / dataset,
        interactive=False,
        subjects=["1"],
        sessions=["a", "b"],
        allow_missing_sessions=allow_missing_sessions,
    )
    # set up the context handler
    context = (
        contextlib.nullcontext()
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
    # The BIDS side of this dataset is ~1.8G, and the pipeline only ever reads it
    # (everything it writes goes under deriv_root, set below), so symlink rather
    # than copy -- copying it does not fit in a typical /tmp tmpfs. The FreeSurfer
    # derivatives below are copied for real, because source/_01_make_bem_surfaces
    # can write BEM surfaces into subjects_dir, which through a symlink would
    # corrupt the user's cached dataset.
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
                    os.symlink(walk_root / _file, dst_dir / offset / bp.basename)
    # emptyroom
    src_dir = config_obj.bids_root / "sub-emptyroom"
    dst_dir = new_bids_path.root / "sub-emptyroom"
    shutil.copytree(src=src_dir, dst=dst_dir, copy_function=os.symlink)
    # root-level files (dataset description, etc)
    src_dir = config_obj.bids_root
    dst_dir = new_bids_path.root
    files = [f for f in src_dir.iterdir() if f.is_file()]
    for _file in files:
        # in theory we should rewrite `participants.tsv` to remove the `sub-02` line,
        # but in practice it will just get ignored so we won't bother.
        os.symlink(_file, dst_dir / _file.name)
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
            / f"sub-01_ses-{sess}_report.h5"
        )
        assert fname.is_file()
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


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param([0, 1, 2], id="0,1,2"),
        pytest.param([3, 4, 5, 6], id="3,4,5,6"),
    ],
)
def test_all_runs_picked(tmp_path: Path, runs: list[str]) -> None:
    """Test that if a task is given, only runs from that task are scanned."""
    dataset = "gh-1140"
    subject = "001"
    task = "FCSRT"
    session = "M0"
    bids_root = tmp_path / dataset
    _make_fake_bids_dataset(
        bids_root,
        subjects=[subject],
        task=task,
        sessions=(session,),
        runs=tuple(f"{r:02d}" for r in runs),
        suffixes=("channels.tsv", "events.tsv", "eeg.vhdr"),
    )
    # fake a config file (can't use static file because `bids_root` is in `tmp_path`)
    config_path = tmp_path / "fake_config_missing_session.py"
    _write_pipeline_config(
        config_path,
        bids_root=bids_root,
        deriv_root=tmp_path / "derivatives" / "mne-bids-pipeline" / dataset,
        subjects=[subject],
        runs="all",
    )
    config = _import_config(config_path=config_path)
    cfg = get_config_data_quality(config=config, subject=subject, session=session)
    ssrt = _get_ssrt(config=config, which=("runs",))
    assert len(ssrt) == len(runs)
    for ri, (this_subject, this_session, this_run, this_task) in enumerate(ssrt):
        assert this_subject == subject
        assert this_session == session
        assert this_run is not None
        assert this_task == task
        assert int(this_run) == runs[ri]
        fnames = get_input_fnames_data_quality(
            cfg=cfg, subject=subject, session=session, run=this_run, task=this_task
        )
        for key, path in fnames.items():
            assert path.fpath.is_file(), f"File for {key=} not found: {path.fpath}"


@pytest.fixture()
def fake_freesurfer_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake a minimal FreeSurfer installation (license + fsaverage dir only)."""
    fs_home = tmp_path / "freesurfer_home"
    (fs_home / "license.txt").parent.mkdir(parents=True, exist_ok=True)
    (fs_home / "license.txt").touch()
    (fs_home / "subjects" / "fsaverage").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FREESURFER_HOME", str(fs_home))
    return fs_home


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_path: Path,
    steps: str,
) -> None:
    monkeypatch.setenv("_MNE_BIDS_STUDY_TESTING", "true")
    monkeypatch.setattr(
        sys, "argv", ["mne_bids_pipeline", str(config_path), f"--steps={steps}"]
    )
    with capsys.disabled():
        print()
        main()


@pytest.mark.parametrize(
    ("on_error", "n_jobs"),
    [
        pytest.param("continue", 1, id="continue-serial"),
        pytest.param("abort", 1, id="abort-serial"),
        pytest.param("debug", 1, id="debug-serial"),
        pytest.param("continue", 2, id="continue-parallel"),
        pytest.param("abort", 2, id="abort-parallel"),
    ],
)
def test_on_error(
    on_error: str,
    n_jobs: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that `on_error` controls whether the pipeline aborts or continues.

    The "-parallel" cases (n_jobs=2) matter because with n_jobs>1 subjects are
    dispatched to separate (loky) worker processes; see _make_fake_bids_dataset.
    """
    subjects = ["01", "02", "03", "04"] if n_jobs > 1 else ["01", "02"]
    task = "task1"
    bids_root = tmp_path / "on_error"
    _make_fake_bids_dataset(bids_root, subjects=subjects, task=task)

    deriv_root = tmp_path / "derivatives" / "mne-bids-pipeline" / "on_error"
    config_path = tmp_path / "config_on_error.py"
    _write_pipeline_config(
        config_path,
        bids_root=bids_root,
        deriv_root=deriv_root,
        subjects=subjects,
        task=task,
        on_error=on_error,
        n_jobs=n_jobs,
    )

    debug_calls: list[None] = []
    if on_error == "debug":
        # Don't actually drop into an interactive debugger.
        monkeypatch.setattr("pdb.post_mortem", lambda tb: debug_calls.append(None))

    steps = "preprocessing/data_quality"
    if on_error == "debug":
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, capsys, config_path, steps)
        assert debug_calls == [None]
    elif on_error == "abort":
        with pytest.raises(Exception):  # noqa: B017
            _run_main(monkeypatch, capsys, config_path, steps)
    else:
        assert on_error == "continue"
        _run_main(monkeypatch, capsys, config_path, steps)
        # Every subject should show up in the log, proving that none were
        # skipped due to an early abort, even under real parallel execution.
        log_files = list(deriv_root.glob("task-*_log.xlsx"))
        assert len(log_files) == 1, log_files
        sheets = pd.read_excel(log_files[0], sheet_name=None)
        (df,) = [df for name, df in sheets.items() if "data_quality" in name]
        seen_subjects = sorted(df["subject"].astype(str).str.zfill(2).unique())
        assert seen_subjects == subjects
        assert (~df["success"]).all()


@pytest.mark.parametrize("on_error", ["continue", "abort"])
def test_recon_all_on_error(
    on_error: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_freesurfer_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that recon-all also honors `on_error` (gh-1022)."""
    subjects = ["01", "02"]
    task = "task1"
    bids_root = tmp_path / "recon_all_on_error"
    _make_fake_bids_dataset(bids_root, subjects=subjects, task=task)

    deriv_root = tmp_path / "derivatives" / "mne-bids-pipeline" / "recon_all_on_error"
    subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
    config_path = tmp_path / "config_recon_all_on_error.py"
    _write_pipeline_config(
        config_path,
        bids_root=bids_root,
        deriv_root=deriv_root,
        subjects=subjects,
        task=task,
        subjects_dir=subjects_dir,
        on_error=on_error,
    )

    # Replace the actual recon-all subprocess call with one that fails
    # unconditionally, and keep track of how many subjects were attempted (this
    # step is not tested under n_jobs>1, so the monkeypatch is reliable here).
    calls: list[None] = []

    def _raise(cmd: list[str], **kwargs: Any) -> None:
        calls.append(None)
        raise RuntimeError("Simulated recon-all failure for on_error test")

    monkeypatch.setattr(_01_recon_all, "run_subprocess", _raise)

    steps = "freesurfer/recon_all"
    if on_error == "abort":
        with pytest.raises(RuntimeError, match="Simulated recon-all failure"):
            _run_main(monkeypatch, capsys, config_path, steps)
    else:
        assert on_error == "continue"
        _run_main(monkeypatch, capsys, config_path, steps)

    # "continue" should process every subject; "abort" should stop as soon as
    # the first subject fails.
    if on_error == "continue":
        assert len(calls) == len(subjects)
    else:
        assert len(calls) == 1


def test_recon_all_skips_existing(
    monkeypatch: pytest.MonkeyPatch,
    fake_freesurfer_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that recon-all is skipped when its output already exists (gh-1022)."""
    subject = "01"
    task = "task1"
    bids_root = tmp_path / "recon_all_skip"
    _make_fake_bids_dataset(bids_root, subjects=[subject], task=task)

    deriv_root = tmp_path / "derivatives" / "mne-bids-pipeline" / "recon_all_skip"
    subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
    # Pre-create the sentinel output file that marks recon-all as already done.
    aseg = subjects_dir / f"sub-{subject}" / "mri" / "aparc+aseg.mgz"
    aseg.parent.mkdir(parents=True, exist_ok=True)
    aseg.touch()

    config_path = tmp_path / "config_recon_all_skip.py"
    _write_pipeline_config(
        config_path,
        bids_root=bids_root,
        deriv_root=deriv_root,
        subjects=[subject],
        task=task,
        subjects_dir=subjects_dir,
    )

    calls: list[None] = []

    def _raise(cmd: list[str], **kwargs: Any) -> None:
        calls.append(None)
        raise RuntimeError("recon-all should not have been invoked")

    monkeypatch.setattr(_01_recon_all, "run_subprocess", _raise)

    _run_main(monkeypatch, capsys, config_path, "freesurfer/recon_all")

    assert calls == []
