"""Test helpers in _config_utils.py."""

import pathlib

from mne_bids_pipeline._config_utils import get_src_fname


def test_get_src_fname(tmp_path: pathlib.Path) -> None:
    """Test that get_src_fname finds source spaces regardless of naming."""
    bem_dir = tmp_path / "sample" / "bem"
    bem_dir.mkdir(parents=True)
    got = get_src_fname(fs_subjects_dir=tmp_path, fs_subject="sample", spacing="oct6")
    want = bem_dir / "sample-oct6-src.fif"
    assert got == want
    assert not got.exists()
    dashed = bem_dir / "sample-oct-6-src.fif"
    dashed.touch()
    got = get_src_fname(fs_subjects_dir=tmp_path, fs_subject="sample", spacing="oct6")
    assert got == dashed
    # If the canonical file also exists, prefer it.
    canonical = bem_dir / "sample-oct6-src.fif"
    canonical.touch()
    got = get_src_fname(fs_subjects_dir=tmp_path, fs_subject="sample", spacing="oct6")
    assert got == canonical
    got = get_src_fname(fs_subjects_dir=tmp_path, fs_subject="sample", spacing=5)
    assert got == bem_dir / "sample-5-src.fif"
    got = get_src_fname(fs_subjects_dir=tmp_path, fs_subject="sample", spacing="all")
    assert got == bem_dir / "sample-all-src.fif"
