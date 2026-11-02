#!/usr/bin/env python
"""Run FreeSurfer's recon-all.

This will run FreeSurfer's ``recon-all --all`` if necessary.
"""

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from mne.utils import run_subprocess

from mne_bids_pipeline._config_utils import (
    _has_session_specific_anat,
    get_fs_subjects_dir,
    get_sessions,
    get_subjects,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run, save_logs
from mne_bids_pipeline.typing import InFilesPathT, OutFilesT

fs_bids_app = Path(__file__).parent / "contrib" / "run.py"


def get_input_fnames_run_recon(
    *, cfg: SimpleNamespace, subject: str, session: str | None
) -> InFilesPathT:
    """Get input filenames for run_recon."""
    # The inputs are the (arbitrarily many) raw anatomical MRI files somewhere in
    # the BIDS dataset, which are impractical to hash individually here; whether to
    # (re)run is instead governed entirely by whether the output below already
    # exists (see get_output_fnames_run_recon).
    return dict()


def get_output_fnames_run_recon(
    *, cfg: SimpleNamespace, subject: str, session: str | None
) -> InFilesPathT:
    out_files = dict()
    subj_dir = cfg.subjects_dir / f"sub-{subject}"
    if session is not None:
        subj_dir = subj_dir.with_name(f"{subj_dir.name}_ses-{session}")
    # aparc+aseg.mgz is one of the last files written by `recon-all -all`, so its
    # presence is a reasonable proxy for "recon-all has already completed".
    out_files["aseg"] = subj_dir / "mri" / "aparc+aseg.mgz"
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_recon,
    get_output_fnames=get_output_fnames_run_recon,
)
def run_recon(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesPathT,
) -> OutFilesT:
    assert not in_files, "run_recon should not receive any input files"
    sub_ses = f"Subject {subject}"
    if session is not None:
        sub_ses = f"{sub_ses} session {session}"
    msg = (
        f"Running recon-all on {sub_ses}. This will take "
        "a LONG time – it's a good idea to let it run over night."
    )
    logger.info(**gen_log_kwargs(message=msg))

    env = os.environ
    if "FREESURFER_HOME" not in env:
        raise RuntimeError("FreeSurfer is not available.")

    license_file = Path(f"{env['FREESURFER_HOME']}/license.txt")
    if not license_file.exists():
        license_file = Path(f"{env['FREESURFER_HOME']}/.license")
    if not license_file.exists():
        raise RuntimeError("FreeSurfer license file not found.")

    cmd = [
        f"{sys.executable}",
        f"{fs_bids_app}",
        f"{cfg.bids_root}",
        f"{cfg.subjects_dir}",
        "participant",
        "--n_cpus=2",
        "--stages=all",
        "--skip_bids_validator",
        f"--license_file={license_file}",
        f"--participant_label={subject}",
    ]
    if session is not None:
        cmd += [f"--session_label={session}"]
    logger.debug("Running: " + " ".join(cmd))
    run_subprocess(cmd, env=env, verbose=logger.level)

    out_files = get_output_fnames_run_recon(cfg=cfg, subject=subject, session=session)
    return _prep_out_files_path(
        exec_params=exec_params,
        out_files=out_files,
        check_relative=cfg.subjects_dir,
    )


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        bids_root=config.bids_root,
        subjects_dir=get_fs_subjects_dir(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run freesurfer recon-all command on BIDS dataset.

    The script allows to run the freesurfer recon-all
    command on all subjects of your BIDS dataset. It can
    run in parallel with the --n_jobs parameter.

    It is built on top of the FreeSurfer BIDS app:

    https://github.com/BIDS-Apps/freesurfer

    and the MNE BIDS Pipeline

    https://mne.tools/mne-bids-pipeline

    You must have freesurfer available on your system.

    Run via the MNE BIDS Pipeline's CLI:

    mne_bids_pipeline --steps=freesurfer --config=your_pipeline_config.py

    """  # noqa
    subjects = get_subjects(config)
    sessions = get_sessions(config)
    cfg = get_config(config=config)
    cfg.subjects_dir.mkdir(parents=True, exist_ok=True)

    # check for session-specific MRIs within subject, and handle accordingly
    subj_sess = list()
    for _subj in subjects:
        for _sess in sessions:
            session = (
                _sess
                if _has_session_specific_anat(_subj, _sess, cfg.subjects_dir)
                else None
            )
            subj_sess.append((_subj, session))

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_recon,
            exec_params=config.exec_params,
            n_iter=len(subj_sess),
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject, session in subj_sess
        )

        # Handle fsaverage
        fsaverage_dir = cfg.subjects_dir / "fsaverage"
        if fsaverage_dir.exists():
            if fsaverage_dir.is_symlink():
                fsaverage_dir.unlink()
            else:
                shutil.rmtree(fsaverage_dir)

        env = os.environ
        shutil.copytree(
            f"{env['FREESURFER_HOME']}/subjects/fsaverage",
            cfg.subjects_dir / "fsaverage",
        )
    save_logs(config=config, logs=logs)
