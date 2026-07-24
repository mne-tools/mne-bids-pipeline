import numpy as np
import mne
import pytest
from types import SimpleNamespace
from mne_bids_pipeline.steps.preprocessing._05b_sync_eyelink import _check_HEOG_ET_vars, _mark_calibration_as_bad
import re
from numpy.polynomial.polynomial import Polynomial
import pandas as pd


def _make_raw(annots, ch_name="eeg1", duration=300.0, sfreq=100.0):
    info = mne.create_info([ch_name], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((1, int(duration * sfreq))), info)
    raw.set_annotations(annots)
    return raw

def test_check_heog_et_vars():
    # SCENARIO 1: 2 heog channels x 1 et channel
    cfg = SimpleNamespace(sync_heog_ch = ("HEOGL", "HEOGR"), sync_et_ch = "xpos_left")
    heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
    assert (heog_ch, et_ch, bipolar) == ("bi_HEOG", ["xpos_left"], True)

    # SCENARIO 2: 1 heog channel x 1 et channel
    cfg = SimpleNamespace(sync_heog_ch = "HEOGL", sync_et_ch = "xpos_left")
    heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
    assert (heog_ch, et_ch, bipolar) == ("HEOGL", ["xpos_left"], False)
    
    # SCENARIO 3: 1 heog channel x 2 et channels
    cfg = SimpleNamespace(sync_heog_ch = "HEOGL", sync_et_ch = ("xpos_left", "xpos_right"))
    heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
    assert (heog_ch, et_ch, bipolar) == ("HEOGL", ["xpos_left", "xpos_right"], False)

    # SCENARIO 4: channels are all None
    cfg = SimpleNamespace(sync_heog_ch = None, sync_et_ch = None)
    heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
    assert (heog_ch, et_ch, bipolar) == (None, [None], False)


def test_mark_calibration_as_bad(capsys):
    def _get_bads(raw):
        bad_recalibration = [(a["description"], a["onset"], a["onset"] + a["duration"]) for a in raw.annotations if a["description"].startswith("BAD_Recalibrate")]
        return bad_recalibration

    cfg = SimpleNamespace(sync_calibration_string = ".* Recalibration (start|end) \\| (.*)")

    # SCENARIO 1: matching star and end
    annots = mne.Annotations([10, 20], [0, 0], ['Note: Recalibration start | 1', 'Note: Recalibration end | 1'])
    raw = _make_raw(annots)
    raw = _mark_calibration_as_bad(raw, cfg)
    bad_recalibration = _get_bads(raw)
    assert bad_recalibration == [('BAD_Recalibrate 1', 10, 20)]

    # SCENARIO 2: repeating starts
    annots = mne.Annotations([10, 20], [0, 0], ['Note: Recalibration start | 1', 'Note: Recalibration start | 1'])
    raw = _make_raw(annots)
    raw = _mark_calibration_as_bad(raw, cfg)
    bad_recalibration = _get_bads(raw)
    captured = capsys.readouterr()
    assert "Encountered apparent duplicate calibration event" in captured.out

    # SCENARIO 3: unmatching start and end
    annots = mne.Annotations([10, 20, 30], [0, 0, 0], ['Note: Recalibration start | 1', 'Note: Recalibration end | 3', 'Note: Recalibration start | 2'])
    raw = _make_raw(annots)
    with pytest.raises(ValueError, match="could not be assigned membership"): _mark_calibration_as_bad(raw, cfg)


def test_sync(capsys):
    #NOTE: chunks copy-pasted from sync_eyelink function. corresponding changes needed if anything is altered

    # check for getting the sync time for eyetracking and eeg data respectively
    def _get_sync_times(raw, raw_et, cfg):
        if not cfg.sync_eventtype_regex_et:
            cfg.sync_eventtype_regex_et = cfg.sync_eventtype_regex
        
        et_sync_times = [annotation["onset"] for annotation in raw_et.annotations if re.search(cfg.sync_eventtype_regex_et,annotation["description"])]
        sync_times    = [annotation["onset"] for annotation in raw.annotations    if re.search(cfg.sync_eventtype_regex,   annotation["description"])]
        assert len(et_sync_times) == len(sync_times),f"Detected eyetracking and EEG sync events were not of equal size ({len(et_sync_times)} vs {len(sync_times)}). Adjust your regular expressions via 'sync_eventtype_regex_et' and 'sync_eventtype_regex' accordingly"
        assert len(sync_times) > 1,f"Not enough distinct sync events for realignment ({len(sync_times)})"

        return et_sync_times, sync_times


    annots_et = mne.Annotations(list(np.arange(1, 301, 1.0)),
                        [0] * 300,
                        ["sync_et"] * 300)
    annots = mne.Annotations(list(np.arange(0, 300, 1.0)),
                        [0] * 300 ,
                        ["sync_eeg"] * 300)

    raw_et = _make_raw(annots_et, ch_name="xpos")
    raw    = _make_raw(annots, ch_name="eeg1")

    ## SCENARIO 1: re exists for both eeg and et signals
    cfg = SimpleNamespace(sync_eventtype_regex = 'sync_eeg', sync_eventtype_regex_et='sync_et')
    et_sync_times, sync_times = _get_sync_times(raw, raw_et, cfg)
    assert (et_sync_times, sync_times) == (list(np.arange(1, 301, 1.0)), list(np.arange(0, 300, 1.0)))

    ## SCENARIO 2: no re defined for et, fall back on the eeg sync re
    cfg = SimpleNamespace(sync_eventtype_regex = 'sync', sync_eventtype_regex_et=None)
    et_sync_times, sync_times = _get_sync_times(raw, raw_et, cfg)
    assert (et_sync_times, sync_times) == (list(np.arange(1, 301, 1.0)), list(np.arange(0, 300, 1.0)))

    ## SCENARIO 3: no re defined for et, fall back onto the eeg sync re. re for eeg does not match et
    cfg = SimpleNamespace(sync_eventtype_regex = 'sync_eeg', sync_eventtype_regex_et=None)
    with pytest.raises(AssertionError, match="Detected eyetracking and EEG sync events were not of equal size"): _get_sync_times(raw, raw_et, cfg)

    ## SCENARIO 4: re not matching
    cfg = SimpleNamespace(sync_eventtype_regex = 'synchronize', sync_eventtype_regex_et=None)
    with pytest.raises(AssertionError, match="Not enough distinct sync events for realignment"): _get_sync_times(raw, raw_et, cfg)


    # Check for alignment
    mne.preprocessing.realign_raw(raw, raw_et, sync_times, et_sync_times)
    raw.add_channels([raw_et], force_update_info=True)
    raw_et.annotations.description = np.array(list(map(lambda desc: "ET_" + desc, raw_et.annotations.description)))
    raw.set_annotations(mne.annotations._combine_annotations(raw.annotations,
                                                                raw_et.annotations,
                                                                0,
                                                                raw.first_samp,
                                                                raw_et.first_samp,
                                                                raw.info["sfreq"]))
    assert "xpos" in raw.ch_names                    
    et_descs = [d for d in raw.annotations.description if d.startswith("ET_")]
    assert len(et_descs) == 300 


    # Check for sync events time
    def _transform(et_df, et_sync_times, sync_times):
        poly = Polynomial.fit(x=et_sync_times, y=sync_times, deg=1)
        converted = poly.convert(domain=(-1, 1))
        [zero_ord, first_ord] = converted.coef
        et_df = et_df.copy()
        et_df["onset"]    = et_df["onset"]    * first_ord + zero_ord
        et_df["end_time"] = et_df["end_time"] * first_ord + zero_ord
        return et_df

    et_sync_times = [0.0, 10.0, 20.0, 30.0]
    sync_times    = [5.0, 25.0, 45.0, 65.0]     #2t + 5
    df = pd.DataFrame({"onset": [1.0, 4.0, 100.0], "end_time": [2.0, 5.0, 101.0]})

    out = _transform(df, et_sync_times, sync_times)
    assert out["onset"].tolist()    == pytest.approx([7.0, 13.0, 205.0])
    assert out["end_time"].tolist() == pytest.approx([9.0, 15.0, 207.0])