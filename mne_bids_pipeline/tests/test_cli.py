"""Test some CLI options."""

import importlib.util
import sys
from pathlib import Path

import pytest

from mne_bids_pipeline._main import main


def test_config_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test the generation of a default config file."""
    cmd = ["mne_bids_pipeline", "--create-config"]
    monkeypatch.setattr(sys, "argv", cmd)
    with pytest.raises(SystemExit, match="2"):
        main()
    cfg_path = tmp_path / "my_config.yaml"
    cmd.append(str(cfg_path))
    main()
    assert cfg_path.is_file()
    spec = importlib.util.spec_from_file_location(str(cfg_path))
    varnames = [v for v in dir(spec) if not v.startswith("__")]
    assert varnames == []
