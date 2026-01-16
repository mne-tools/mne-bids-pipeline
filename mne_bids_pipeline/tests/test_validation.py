"""Test the pipeline configuration import validator."""

from pathlib import Path
from shutil import rmtree

import pytest

from mne_bids_pipeline._config_import import ConfigError, _import_config


def test_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that misspellings are caught by our config import validator."""
    config_path = tmp_path / "config.py"
    bad_text = ""
    # no bids_root
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match="You need to specify `bids_root`"):
        _import_config(config_path=config_path)
    bad_text += f"bids_root = '{tmp_path}'\n"
    # no ch_types
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match="Value should have at least 1 item"):
        _import_config(config_path=config_path)
    bad_text += "ch_types = ['eeg']\n"
    # conditions
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match="name of your conditions"):
        _import_config(config_path=config_path)
    bad_text += "conditions = ['foo']\n"
    config_path.write_text(bad_text)
    _import_config(config_path=config_path)  # working
    working_text = bad_text
    # misspelled sessions
    bad_text += "session = ['foo']\n"
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match=r".*did you mean 'sessions'\?"):
        _import_config(config_path=config_path)
    bad_text += "config_validation = 'warn'\n"
    config_path.write_text(bad_text)
    capsys.readouterr()
    _import_config(config_path=config_path)
    msg, err = capsys.readouterr()
    assert err == ""
    assert len(msg.splitlines()) == 1
    assert "did you mean 'sessions'?" in msg
    bad_text += "config_validation = 'ignore'\n"
    config_path.write_text(bad_text)
    capsys.readouterr()
    _import_config(config_path=config_path)
    msg, err = capsys.readouterr()
    assert msg == err == ""  # no new message
    # TWA headpos without movement compensation
    bad_text = working_text + "mf_destination = 'twa'\n"
    config_path.write_text(bad_text)
    with pytest.raises(ConfigError, match="cannot compute time-weighted average head"):
        _import_config(config_path=config_path)
    # maxfilter extra kwargs
    bad_text = working_text + "mf_extra_kws = {'calibration': 'x', 'head_pos': False}\n"
    config_path.write_text(bad_text)
    with pytest.raises(ConfigError, match="contains keys calibration, head_pos that"):
        _import_config(config_path=config_path)
    # ecg_channel_dict key validation (all subjects have channels specified)
    try:
        # these must exist for dict check to work
        for sub, ses in {"1": "a", "2": "b"}.items():
            _dir = tmp_path / f"sub-{sub}" / f"ses-{ses}" / "eeg"
            _dir.mkdir(parents=True)
            (_dir / f"sub-{sub}_ses-{ses}_eeg.fif").touch()
    except Exception:
        raise
    else:
        # test the config import when sessions = "all"  (default)
        bad_text = (
            working_text + "subjects = ['1', '2']\n"
            "allow_missing_sessions = True\n"
            "ssp_ecg_channel = {'sub-1': 'MEG0111'}\n"
        )  # OK to omit session from sub-1, but entry for sub-2 is missing
        config_path.write_text(bad_text)
        with pytest.raises(ConfigError, match=r"Missing entries.*\n  sub-2_ses-b"):
            _import_config(config_path=config_path)
        # test when single session specified in config
        bad_text = (
            working_text + "subjects = ['1', '2']\n"
            "sessions = ['a']\n"
            "allow_missing_sessions = True\n"
            "ssp_ecg_channel = {'sub-1_ses-b': 'MEG0111'}\n"  # no entry for sub-1_ses-a
        )
        config_path.write_text(bad_text)
        with pytest.raises(ConfigError, match=r"Missing entries.*\n  sub-1_ses-a"):
            _import_config(config_path=config_path)
    # clean up
    finally:
        for sub in ("1", "2"):
            rmtree(tmp_path / f"sub-{sub}")

    # ecg_channel_dict key validation (keys in dict are well-formed)
    bad_text = working_text + "ssp_ecg_channel = {'sub-0_1': 'MEG0111'}\n"  # underscore
    config_path.write_text(bad_text)
    with pytest.raises(ConfigError, match="Malformed keys in ssp_ecg_channel dict:.*"):
        _import_config(config_path=config_path)
    # old values
    bad_text = working_text
    bad_text += "debug = True\n"
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match="Found a variable.*use on_error=.*"):
        _import_config(config_path=config_path)
    bad_text += "on_error = 'debug' if debug else 'raise'\n"
    config_path.write_text(bad_text)
    _import_config(config_path=config_path)  # this is okay
    # matching deriv and bids
    bad_text = working_text
    bad_text += f"deriv_root = '{tmp_path}'"
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match="cannot be the same"):
        _import_config(config_path=config_path)
