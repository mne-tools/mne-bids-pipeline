"""Test EEG referencing."""

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from mne_bids_pipeline._reference import _set_eeg_reference


def _make_raw() -> mne.io.RawArray:
    info = mne.create_info(["Fz", "Cz", "Pz", "MEG0111"], 100.0, ["eeg"] * 3 + ["mag"])
    data = np.random.default_rng(0).standard_normal((4, 100))
    return mne.io.RawArray(data, info, verbose="error")


@pytest.mark.parametrize("online_ref", [None, "M1"])
@pytest.mark.parametrize("apply", [False, True])
def test_set_eeg_reference(online_ref: str | None, apply: bool) -> None:
    """Test that the online reference is included in the average."""
    cfg = SimpleNamespace(
        ch_types=["eeg"],
        eeg_reference="average",
        eeg_online_reference_channel=online_ref,
    )
    raw = _make_raw()
    eeg = np.asarray(raw.get_data(picks="eeg"))
    n_avg = len(eeg) + (online_ref is not None)
    want = eeg - eeg.sum(axis=0) / n_avg
    _set_eeg_reference(raw, cfg=cfg, apply=apply)
    added = online_ref is not None and not apply
    assert len(raw.ch_names) == 4 + added
    raw.apply_proj()
    np.testing.assert_allclose(raw.get_data()[:3], want)  # ref is appended
    # idempotent once the projector exists
    _set_eeg_reference(raw, cfg=cfg, apply=apply)
    assert len(raw.ch_names) == 4 + added

    # channel reference, and nothing to do without EEG
    cfg.eeg_reference = ["Cz"]
    raw = _make_raw()
    _set_eeg_reference(raw, cfg=cfg)
    assert raw.ch_names == _make_raw().ch_names
    np.testing.assert_allclose(raw.get_data(picks="Cz"), 0.0)
    raw = _make_raw().pick("mag")
    _set_eeg_reference(raw, cfg=cfg)
