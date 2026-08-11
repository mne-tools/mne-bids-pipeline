"""Test recording of the files that pipeline steps read and write."""

import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
from mne_bids import BIDSPath

from mne_bids_pipeline._flow import (
    FlowEntryT,
    _flow_files,
    _read_flow_entries,
    _write_flow_entry,
)
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def _entry(
    step: str,
    *,
    func: str = "func",
    subject: str | None = "01",
    session: str | None = None,
    run: str | None = None,
    task: str | None = "av",
    in_files: dict[str, str] | None = None,
    out_files: dict[str, str] | None = None,
) -> FlowEntryT:
    return {
        "step": step,
        "func": func,
        "subject": subject,
        "session": session,
        "run": run,
        "task": task,
        "in_files": in_files or dict(),
        "out_files": out_files or dict(),
    }


def _pipeline_entries() -> list[FlowEntryT]:
    """Get a small but representative recording: 3 runs, a fan-out, a dead end."""
    entries = list()
    for run in ("01", "02", "03"):
        entries.append(
            _entry(
                "preprocessing/_04_frequency_filter",
                func="filter_data",
                run=run,
                in_files={"raw": f"/bids/sub-01_task-av_run-{run}_meg.fif"},
                out_files={"raw": f"/deriv/sub-01_task-av_run-{run}_proc-filt_raw.fif"},
            )
        )
    entries.append(
        _entry(
            "preprocessing/_07_make_epochs",
            func="make_epochs",
            in_files={
                f"raw_{run}": f"/deriv/sub-01_task-av_run-{run}_proc-filt_raw.fif"
                for run in ("01", "02", "03")
            },
            out_files={"epo": "/deriv/sub-01_task-av_epo.fif"},
        )
    )
    for step in ("sensor/_01_make_evoked", "sensor/_06_make_cov"):
        entries.append(
            _entry(
                step,
                in_files={"epo": "/deriv/sub-01_task-av_epo.fif"},
                out_files={"out": f"/deriv/sub-01_task-av_{step[-3:]}.fif"},
            )
        )
    # Produces something nobody reads, so it should not show up at all
    entries.append(
        _entry(
            "init/_01_init_derivatives_dir",
            func="init_dataset",
            subject=None,
            task=None,
            out_files={"json": "/deriv/dataset_description.json"},
        )
    )
    return entries


def test_flow_files() -> None:
    """Test normalization of the in_files/out_files mappings."""
    bids_path = BIDSPath(
        subject="01",
        root="/bids",
        datatype="meg",
        suffix="meg",
        extension=".fif",
        check=False,
    )
    got = _flow_files(
        {
            "path": pathlib.Path("/deriv/a.fif"),
            "bids": bids_path,
            "hashed": ("/deriv/b.fif", 1234.0),
            "__unknown_inputs__": "custom cov",
            "junk": None,
        }
    )
    assert got == {
        "path": "/deriv/a.fif",
        "bids": str(bids_path.fpath),
        "hashed": "/deriv/b.fif",
    }


def test_flow_recording_roundtrip(tmp_path: pathlib.Path) -> None:
    """Test that recorded entries survive a write/read cycle without duplicating."""
    entries = _pipeline_entries()
    for entry in entries:
        _write_flow_entry(deriv_root=tmp_path, entry=entry)
    # Re-running a step must overwrite its own entry rather than add another
    changed = _entry(
        entries[0]["step"],
        func=entries[0]["func"],
        run=entries[0]["run"],
        in_files=entries[0]["in_files"],
        out_files={"raw": "/deriv/other.fif"},
    )
    _write_flow_entry(deriv_root=tmp_path, entry=changed)

    got = _read_flow_entries(deriv_root=tmp_path, subject="01", session=None)
    assert len(got) == len(entries)
    assert changed in got
    assert entries[0] not in got
    # The dataset-level entry lives in its own file but is relevant to every subject
    assert sum(entry["subject"] is None for entry in got) == 1
    assert (tmp_path / ".pipeline_flow" / "dataset.json").is_file()
    assert (tmp_path / ".pipeline_flow" / "sub-01.json").is_file()

    dataset_only = [entry for entry in got if entry["subject"] is None]
    assert _read_flow_entries(deriv_root=tmp_path, subject="02", session=None) == (
        dataset_only
    )
    # Entries of other sessions are filtered out
    other = _entry("sensor/_01_make_evoked", session="t2")
    _write_flow_entry(deriv_root=tmp_path, entry=other)
    assert other not in _read_flow_entries(
        deriv_root=tmp_path, subject="01", session="t1"
    )
    assert other in _read_flow_entries(deriv_root=tmp_path, subject="01", session="t2")


def test_flow_recording_missing(tmp_path: pathlib.Path) -> None:
    """Test that an absent or unreadable recording is not fatal."""
    assert _read_flow_entries(deriv_root=tmp_path, subject="01", session=None) == []
    fname = tmp_path / ".pipeline_flow" / "sub-01.json"
    fname.parent.mkdir()
    fname.write_text("not json")
    assert _read_flow_entries(deriv_root=tmp_path, subject="01", session=None) == []


_N_CALLS: list[str] = list()


def _get_input_fnames_flow(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    return dict(raw=cfg.raw)


def _get_output_fnames_flow(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    return dict(filt=cfg.out)


def _flow_step_impl(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    _N_CALLS.append(subject)
    in_files.pop("raw")
    cfg.out.write_text("filtered")
    return _prep_out_files_path(exec_params=exec_params, out_files=dict(filt=cfg.out))


_flow_step = failsafe_run(get_input_fnames=_get_input_fnames_flow)(_flow_step_impl)
_flow_step_out = failsafe_run(
    get_input_fnames=_get_input_fnames_flow,
    get_output_fnames=_get_output_fnames_flow,
)(_flow_step_impl)


@pytest.fixture
def flow_kwargs(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Get kwargs for a fake pipeline step writing into a fresh derivatives dir."""
    _N_CALLS.clear()
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir()
    raw = tmp_path / "sub-01_task-av_run-01_meg.fif"
    raw.write_text("raw")
    cfg = SimpleNamespace(
        raw=raw, out=deriv_root / "sub-01_task-av_run-01_filt_raw.fif"
    )
    exec_params = SimpleNamespace(
        on_error="abort",
        deriv_root=deriv_root,
        memory_location=True,
        memory_subdir="joblib",
        memory_verbose=0,
        memory_file_method="mtime",
        ignore_warnings=(),
    )
    return dict(
        cfg=cfg,
        exec_params=exec_params,
        subject="01",
        session=None,
        run="01",
        task="av",
    )


def _recorded(flow_kwargs: dict[str, Any]) -> list[FlowEntryT]:
    deriv_root = flow_kwargs["exec_params"].deriv_root
    return _read_flow_entries(deriv_root=deriv_root, subject="01", session=None)


@pytest.mark.parametrize("memory_location", (True, False))
def test_flow_recorder(flow_kwargs: dict[str, Any], memory_location: bool) -> None:
    """Test that the step wrapper records files on fresh runs and on cache hits."""
    flow_kwargs["exec_params"].memory_location = memory_location
    cfg = flow_kwargs["cfg"]
    _flow_step(**flow_kwargs)
    want = [
        {
            "step": "tests/test_flow",
            "func": "_flow_step_impl",
            "subject": "01",
            "session": None,
            "run": "01",
            "task": "av",
            "in_files": {"raw": str(cfg.raw)},
            "out_files": {"filt": str(cfg.out)},
        }
    ]
    assert _recorded(flow_kwargs) == want
    _flow_step(**flow_kwargs)
    assert len(_N_CALLS) == (1 if memory_location else 2)  # cache hit the 2nd time
    assert _recorded(flow_kwargs) == want  # ... and still recorded


def test_flow_recorder_short_circuit(flow_kwargs: dict[str, Any]) -> None:
    """Test that a step skipped because its outputs exist is still recorded."""
    flow_kwargs["cfg"].out.write_text("stale")
    _flow_step_out(**flow_kwargs)
    assert _N_CALLS == []
    (entry,) = _recorded(flow_kwargs)
    assert entry["in_files"] == {"raw": str(flow_kwargs["cfg"].raw)}
    assert entry["out_files"] == {"filt": str(flow_kwargs["cfg"].out)}


def test_flow_recorder_failure(
    flow_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a broken recording does not take the pipeline down with it."""

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("no disk for you")

    monkeypatch.setattr("mne_bids_pipeline._run._write_flow_entry", _boom)
    _flow_step(**flow_kwargs)
    assert _N_CALLS == ["01"]
    assert _recorded(flow_kwargs) == []
