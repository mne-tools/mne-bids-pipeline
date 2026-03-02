#!/usr/bin/env python
"""Generate coregistration surfaces.

Use FreeSurfer's ``mkheadsurf`` and related utilities to make head surfaces
suitable for coregistration.
"""

from pathlib import Path
from types import SimpleNamespace

import mne.bem

from mne_bids_pipeline._config_utils import (
    _get_ss,
    get_fs_subject,
    get_fs_subjects_dir,
    get_sessions,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run
from mne_bids_pipeline.typing import InFilesPathT, OutFilesT

fs_bids_app = Path(__file__).parent / "contrib" / "run.py"


def _get_scalp_in_files(cfg: SimpleNamespace) -> InFilesPathT:
    subject_path = Path(cfg.fs_subjects_dir) / cfg.fs_subject
    seghead = subject_path / "surf" / "lh.seghead"
    in_files = dict()
    if seghead.is_file():
        in_files["seghead"] = seghead
    else:
        in_files["t1"] = subject_path / "mri" / "T1.mgz"
    return in_files


def get_input_fnames_coreg_surfaces(
    *,
    cfg: SimpleNamespace,
    subject: str,
) -> InFilesPathT:
    return _get_scalp_in_files(cfg)


def get_output_fnames_coreg_surfaces(
    *, cfg: SimpleNamespace, subject: str
) -> InFilesPathT:
    out_files = dict()
    subject_path = Path(cfg.fs_subjects_dir) / cfg.fs_subject
    out_files["seghead"] = subject_path / "surf" / "lh.seghead"
    for key in ("dense", "medium", "sparse"):
        out_files[f"head-{key}"] = (
            subject_path / "bem" / f"{cfg.fs_subject}-head-{key}.fif"
        )
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_coreg_surfaces,
    get_output_fnames=get_output_fnames_coreg_surfaces,
)
def make_coreg_surfaces(
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    in_files: InFilesPathT,
) -> OutFilesT:
    """Create head surfaces for use with MNE-Python coregistration tools."""
    msg = "Creating scalp surfaces for coregistration"
    logger.info(**gen_log_kwargs(message=msg))
    in_files.pop("t1" if "t1" in in_files else "seghead")
    mne.bem.make_scalp_surfaces(
        subject=cfg.fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        force=True,
        overwrite=True,
    )
    out_files = get_output_fnames_coreg_surfaces(cfg=cfg, subject=subject)
    return _prep_out_files_path(
        exec_params=exec_params,
        out_files=out_files,
        check_relative=cfg.fs_subjects_dir,
    )


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=get_fs_subject(config, subject, session=session),
        fs_subjects_dir=get_fs_subjects_dir(config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    # Ensure we're also processing fsaverage if present
    ss = _get_ss(config=config)
    sessions = get_sessions(config)
    if (Path(get_fs_subjects_dir(config)) / "fsaverage").exists():
        ss += [("fsaverage", session) for session in sessions]
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            make_coreg_surfaces,
            exec_params=config.exec_params,
            n_iter=len(ss),
        )

        parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session,
                ),
                exec_params=config.exec_params,
                force_run=config.recreate_scalp_surface,
                subject=subject,
            )
            for subject, session in ss
        )
