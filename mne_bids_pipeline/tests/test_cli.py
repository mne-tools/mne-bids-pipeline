"""Test some CLI options."""

import importlib
import sys
import pytest
from mne_bids_pipeline._main import main


def test_config_generation(tmp_path, monkeypatch):
    cmd = ['mne_bids_pipeline', '--create-config']
    monkeypatch.setattr(sys, 'argv', cmd)
    with pytest.raises(SystemExit, match='2'):
        main()
    cfg_path = tmp_path / 'my_config.yaml'
    cmd.append(str(cfg_path))
    main()
    assert cfg_path.is_file()
    spec = importlib.util.spec_from_file_location(cfg_path)
    varnames = [v for v in dir(spec) if not v.startswith('__')]
    assert varnames == []
