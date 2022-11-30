import pytest
from mne_bids_pipeline._config_import import _import_config


def test_validation(tmp_path, caplog):
    """Test that misspellings are caught by our config import validator."""
    config_path = tmp_path / 'config.py'
    bad_text = ''
    # no bids_root
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match='You need to specify `bids_root`'):
        _import_config(config_path=config_path)
    bad_text += f"bids_root = '{tmp_path}'\n"
    # no ch_types
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match='Please specify ch_types'):
        _import_config(config_path=config_path)
    bad_text += "ch_types = ['eeg']\n"
    # conditions
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match='name of your conditions'):
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
    assert caplog.record_tuples == []
    _import_config(config_path=config_path)
    assert len(caplog.record_tuples) == 1
    assert caplog.record_tuples[0][:2] == ('mne-bids-pipeline', 30)
    msg = caplog.record_tuples[0][2]
    assert "did you mean 'sessions'?" in msg
    bad_text += "config_validation = 'ignore'\n"
    config_path.write_text(bad_text)
    _import_config(config_path=config_path)
    assert len(caplog.record_tuples) == 1  # no new message
    # old values
    bad_text = working_text
    bad_text += "debug = True\n"
    config_path.write_text(bad_text)
    with pytest.raises(ValueError, match='Found a variable.*use on_error=.*'):
        _import_config(config_path=config_path)
    bad_text += "on_error = 'debug' if debug else 'raise'\n"
    config_path.write_text(bad_text)
    _import_config(config_path=config_path)  # this is okay
