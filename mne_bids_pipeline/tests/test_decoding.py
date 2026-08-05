"""Tests for decoding edge cases."""

from contextlib import contextmanager
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath
from scipy.io import loadmat, savemat

from mne_bids_pipeline import _decoding
from mne_bids_pipeline.steps.sensor import (
    _02_decoding_full_epochs as full_epochs,
)
from mne_bids_pipeline.steps.sensor import (
    _03_decoding_time_by_time as time_by_time,
)
from mne_bids_pipeline.steps.sensor import (
    _99_group_average as group_average,
)


def test_insufficient_decoding_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insufficient class counts should produce predictable NaN arrays."""
    messages = []
    monkeypatch.setattr(
        _decoding.logger,
        "warning",
        lambda msg, **kwargs: messages.append(msg),
    )

    full_epochs_scores = _decoding._get_nan_decoding_scores_if_insufficient(
        n_cond1=2,
        n_cond2=12,
        n_splits=5,
        score_shape=(),
        contrast_msg="A – B",
    )
    assert full_epochs_scores is not None
    assert full_epochs_scores.shape == (5,)
    assert np.isnan(full_epochs_scores).all()

    time_by_time_scores = _decoding._get_nan_decoding_scores_if_insufficient(
        n_cond1=2,
        n_cond2=12,
        n_splits=5,
        score_shape=(7,),
        contrast_msg="A – B",
    )
    assert time_by_time_scores is not None
    assert time_by_time_scores.shape == (5, 7)
    assert np.isnan(time_by_time_scores).all()
    assert "found 2 and 12" in messages[0]

    generalization_scores = _decoding._get_nan_decoding_scores_if_insufficient(
        n_cond1=2,
        n_cond2=12,
        n_splits=5,
        score_shape=(7, 7),
        contrast_msg="A – B",
    )
    assert generalization_scores is not None
    assert generalization_scores.shape == (5, 7, 7)
    assert np.isnan(generalization_scores).all()

    sufficient_scores = _decoding._get_nan_decoding_scores_if_insufficient(
        n_cond1=5,
        n_cond2=12,
        n_splits=5,
        score_shape=(),
        contrast_msg="A – B",
    )
    assert sufficient_scores is None


def _subject_cfg(tmp_path, *, time_generalization: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        conditions={"A": "A", "B": "B"},
        contrasts=[["A", "B"]],
        decoding_epochs_tmax=0.1,
        decoding_epochs_tmin=-0.1,
        decoding_metric="roc_auc",
        decoding_n_splits=5,
        decoding_time_decim=1,
        decoding_time_generalization=time_generalization,
        decoding_time_generalization_decim=1,
        deriv_root=tmp_path,
        random_state=42,
    )


def _epochs_with_insufficient_class() -> mne.EpochsArray:
    info = mne.create_info(["EEG001"], sfreq=50, ch_types="eeg")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((14, 1, 11))
    events = np.column_stack(
        [
            np.arange(14) * 20,
            np.zeros(14, dtype=int),
            np.r_[np.ones(2, dtype=int), np.full(12, 2, dtype=int)],
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"A": 1, "B": 2},
        tmin=-0.1,
        verbose="error",
    )


class _Report:
    def add_figure(self, **kwargs) -> None:
        pass


@contextmanager
def _open_report(**kwargs):
    yield _Report()


def _input_epochs_path(tmp_path) -> BIDSPath:
    path = BIDSPath(
        root=tmp_path,
        subject="01",
        task="test",
        datatype="eeg",
        suffix="epo",
        extension=".fif",
        check=False,
    )
    path.fpath.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_run_full_epochs_writes_nan_scores_when_epochs_are_insufficient(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full-epoch step should skip fitting and save all-NaN folds."""
    cfg = _subject_cfg(tmp_path)
    epochs = _epochs_with_insufficient_class()
    monkeypatch.setattr(full_epochs.mne, "read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr(
        full_epochs, "_restrict_analyze_channels", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        full_epochs,
        "_decoding_preproc_steps",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("preprocessing should not be constructed")
        ),
    )
    monkeypatch.setattr(
        full_epochs,
        "cross_val_score",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cross-validation should not run")
        ),
    )
    monkeypatch.setattr(full_epochs, "_open_report", _open_report)
    monkeypatch.setattr(full_epochs, "_get_prefix_tags", lambda **kwargs: ("", ()))
    monkeypatch.setattr(
        full_epochs,
        "_plot_full_epochs_decoding_scores",
        lambda **kwargs: (plt.figure(), "", pd.DataFrame()),
    )
    monkeypatch.setattr(
        full_epochs,
        "_prep_out_files",
        lambda *, out_files, exec_params: out_files,
    )

    out_files = full_epochs.run_epochs_decoding.__wrapped__(
        cfg=cfg,
        exec_params=SimpleNamespace(),
        subject="01",
        session=None,
        task="test",
        condition1="A",
        condition2="B",
        in_files={"epochs": _input_epochs_path(tmp_path)},
    )

    result = loadmat(
        next(path for key, path in out_files.items() if key.startswith("mat"))
    )
    assert result["scores"].shape == (1, 5)
    assert np.isnan(result["scores"]).all()


def test_run_time_by_time_writes_nan_scores_when_epochs_are_insufficient(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The time-resolved step should skip fitting and preserve output shape."""
    cfg = _subject_cfg(tmp_path)
    epochs = _epochs_with_insufficient_class()
    monkeypatch.setattr(time_by_time.mne, "read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr(
        time_by_time, "_restrict_analyze_channels", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        time_by_time,
        "_decoding_preproc_steps",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("preprocessing should not be constructed")
        ),
    )
    monkeypatch.setattr(
        time_by_time,
        "cross_val_multiscore",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cross-validation should not run")
        ),
    )
    monkeypatch.setattr(time_by_time, "_open_report", _open_report)
    monkeypatch.setattr(time_by_time, "_get_prefix_tags", lambda **kwargs: ("", ()))
    monkeypatch.setattr(
        time_by_time,
        "get_parallel_backend",
        lambda exec_params: _open_report(),
    )
    monkeypatch.setattr(
        time_by_time,
        "get_parallel_backend_name",
        lambda exec_params: "loky",
    )
    monkeypatch.setattr(
        time_by_time,
        "_plot_time_by_time_decoding_scores",
        lambda **kwargs: plt.figure(),
    )
    monkeypatch.setattr(
        time_by_time,
        "_prep_out_files",
        lambda *, out_files, exec_params: out_files,
    )

    out_files = time_by_time.run_time_decoding.__wrapped__(
        cfg=cfg,
        exec_params=SimpleNamespace(n_jobs=1),
        subject="01",
        session=None,
        task="test",
        condition1="A",
        condition2="B",
        in_files={"epochs": _input_epochs_path(tmp_path)},
    )

    result = loadmat(
        next(path for key, path in out_files.items() if key.startswith("mat"))
    )
    assert result["scores"].shape == (5, len(epochs.times))
    assert np.isnan(result["scores"]).all()


def test_exclude_all_nan_decoding_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only subjects whose entire decoding result is NaN should be excluded."""
    messages = []
    monkeypatch.setattr(
        group_average.logger,
        "warning",
        lambda msg, **kwargs: messages.append(msg),
    )
    mean_scores = np.array(
        [
            [0.6, 0.7],
            [np.nan, np.nan],
            [np.nan, 0.8],
        ]
    )

    scores, subjects = group_average._exclude_all_nan_decoding_subjects(
        mean_scores=mean_scores,
        subjects=["01", "02", "03"],
        contrast_msg="A – B",
    )

    assert subjects == ["01", "03"]
    assert np.array_equal(scores, mean_scores[[0, 2]], equal_nan=True)
    assert "02" in messages[0]
    assert "03" not in messages[0]


def _group_cfg(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        acq=None,
        cluster_forming_t_threshold=None,
        cluster_n_permutations=16,
        cluster_permutation_p_threshold=0.05,
        datatype="eeg",
        decoding_metric="roc_auc",
        decoding_time_decim=1,
        decoding_time_generalization=False,
        decoding_time_generalization_decim=1,
        deriv_root=tmp_path,
        n_boot=32,
        random_state=42,
        rec=None,
        space=None,
        subjects=["01", "02", "03"],
    )


def test_average_full_epochs_uses_effective_n(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-NaN subject results should not enter full-epoch group statistics."""
    cfg = _group_cfg(tmp_path)
    monkeypatch.setattr(
        group_average,
        "get_subjects_given_session",
        lambda cfg, session: ["01", "02", "03"],
    )
    monkeypatch.setattr(
        group_average,
        "_prep_out_files",
        lambda *, out_files, exec_params: out_files,
    )

    in_files = {"epochs": tmp_path / "unused-epo.fif"}
    for subject, scores in {
        "01": np.array([0.6, 0.7]),
        "02": np.array([np.nan, np.nan]),
        "03": np.array([0.7, 0.8]),
    }.items():
        fname = tmp_path / f"sub-{subject}-scores.mat"
        savemat(fname, {"scores": scores})
        in_files[f"scores-{subject}"] = fname

    out_files = group_average.average_full_epochs_decoding.__wrapped__(
        cfg=cfg,
        exec_params=SimpleNamespace(),
        subject="average",
        session=None,
        cond_1="A",
        cond_2="B",
        task="test",
        in_files=in_files,
    )
    result = loadmat(out_files["mat"])

    assert result["N"].item() == 2
    assert np.allclose(result["scores"].squeeze(), [0.65, 0.75])
    assert result["mean"].item() == pytest.approx(0.7)


@pytest.mark.parametrize("expected_n", [0, 1])
def test_average_time_by_time_skips_permutation_below_effective_n_two(
    tmp_path, monkeypatch: pytest.MonkeyPatch, expected_n: int
) -> None:
    """Permutation testing and statistics should use the effective N."""
    cfg = _group_cfg(tmp_path)
    cfg.subjects = ["01", "02"]
    times = np.array([-0.1, 0.0, 0.1])

    class _Epochs:
        def __init__(self) -> None:
            self.times = times

        def decimate(self, decim, verbose):
            raise AssertionError("decimate should not be called for decim=1")

    class _Report:
        def add_figure(self, **kwargs) -> None:
            pass

    @contextmanager
    def _open_report(**kwargs):
        yield _Report()

    monkeypatch.setattr(
        group_average.mne, "read_epochs", lambda *args, **kwargs: _Epochs()
    )
    monkeypatch.setattr(
        group_average,
        "get_subjects_given_session",
        lambda cfg, session: ["01", "02"],
    )
    monkeypatch.setattr(group_average, "_open_report", _open_report)
    monkeypatch.setattr(group_average, "_get_prefix_tags", lambda **kwargs: ("", ()))
    monkeypatch.setattr(
        group_average,
        "_plot_time_by_time_decoding_scores_gavg",
        lambda **kwargs: plt.figure(),
    )
    monkeypatch.setattr(
        group_average,
        "_decoding_cluster_permutation_test",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("permutation test should not run")
        ),
    )
    monkeypatch.setattr(
        group_average,
        "_prep_out_files",
        lambda *, out_files, exec_params: out_files,
    )

    output_path = group_average._decoding_out_fname(
        cfg=cfg,
        subject="average",
        session=None,
        cond_1="A",
        cond_2="B",
        task="test",
        kind="TimeByTime",
    )
    output_path.fpath.parent.mkdir(parents=True, exist_ok=True)

    valid_scores = np.array(
        [
            [0.5, 0.6, 0.7],
            [0.6, 0.7, 0.8],
        ]
    )
    if expected_n == 0:
        valid_scores[:] = np.nan
    in_files = {"epochs": tmp_path / "unused-epo.fif"}
    for subject, scores in {
        "01": valid_scores,
        "02": np.full((2, 3), np.nan),
    }.items():
        fname = tmp_path / f"sub-{subject}-scores.mat"
        savemat(fname, {"scores": scores})
        in_files[f"scores-{subject}"] = fname

    out_files = group_average.average_time_by_time_decoding.__wrapped__(
        cfg=cfg,
        exec_params=SimpleNamespace(),
        subject="average",
        session=None,
        task="test",
        cond_1="A",
        cond_2="B",
        in_files=in_files,
    )
    result = loadmat(out_files["mat"])

    assert result["N"].item() == expected_n
    if expected_n:
        assert np.allclose(result["mean"].squeeze(), [0.55, 0.65, 0.75])
    else:
        assert np.isnan(result["mean"]).all()
    assert np.isnan(result["cluster_n_permutations"]).all()


def test_plot_full_epochs_handles_contrast_specific_effective_n() -> None:
    """The grand-average plot should not assume equal N across contrasts."""
    fig, caption, data = group_average._plot_full_epochs_decoding_scores(
        contrast_names=["A vs. B", "C vs. D"],
        scores=[np.array([0.6, 0.7]), np.array([0.8])],
        metric="roc_auc",
        kind="grand-average",
    )
    try:
        assert data.groupby("Contrast").size().to_dict() == {
            "A vs. B": 2,
            "C vs. D": 1,
        }
        assert "A vs. B: N=2" in caption
        assert "C vs. D: N=1" in caption
    finally:
        plt.close(fig)
