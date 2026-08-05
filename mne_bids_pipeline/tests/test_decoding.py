"""Tests for decoding edge cases."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_bids_pipeline import _decoding
from mne_bids_pipeline.steps.sensor import _99_group_average as group_average


@pytest.mark.parametrize("score_shape", [(), (7,), (7, 7)])
def test_insufficient_decoding_epochs(
    score_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insufficient class counts should produce predictable NaN arrays."""
    messages = []
    monkeypatch.setattr(
        _decoding.logger, "warning", lambda msg, **kwargs: messages.append(msg)
    )

    scores = _decoding._get_nan_decoding_scores_if_insufficient(
        n_cond1=2,
        n_cond2=12,
        n_splits=5,
        score_shape=score_shape,
        contrast_msg="A – B",
    )

    assert scores is not None
    assert scores.shape == (5, *score_shape)
    assert np.isnan(scores).all()
    assert "found 2 and 12" in messages[0]

    assert (
        _decoding._get_nan_decoding_scores_if_insufficient(
            n_cond1=5,
            n_cond2=12,
            n_splits=5,
            score_shape=score_shape,
            contrast_msg="A – B",
        )
        is None
    )


def test_exclude_all_nan_decoding_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only subjects whose entire decoding result is NaN should be excluded."""
    messages = []
    monkeypatch.setattr(
        group_average.logger, "warning", lambda msg, **kwargs: messages.append(msg)
    )
    mean_scores = np.array([[0.6, 0.7], [np.nan, np.nan], [np.nan, 0.8]])

    scores, subjects = group_average._exclude_all_nan_decoding_subjects(
        mean_scores=mean_scores,
        subjects=["01", "02", "03"],
        contrast_msg="A – B",
    )

    assert subjects == ["01", "03"]
    assert np.array_equal(scores, mean_scores[[0, 2]], equal_nan=True)
    assert "02" in messages[0]
    assert "03" not in messages[0]


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
