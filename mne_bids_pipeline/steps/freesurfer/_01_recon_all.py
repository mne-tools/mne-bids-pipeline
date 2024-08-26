#!/usr/bin/env python
"""Run FreeSurfer's recon-all.

This will run FreeSurfer's ``recon-all --all`` if necessary.
"""

import os
import shutil
import sys
from pathlib import Path

from mne.utils import run_subprocess

from mne_bids_pipeline._config_utils import (
    get_fs_subjects_dir,
    get_sessions,
    get_subjects,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func

fs_bids_app = Path(__file__).parent / "contrib" / "run.py"


def run_recon(root_dir, subject, fs_bids_app, subjects_dir, session=None) -> None:
    subj_dir = subjects_dir / f"sub-{subject}"
    sub_ses = f"Subject {subject}"
    if session is not None:
        subj_dir = subj_dir / f"ses-{session}"
        sub_ses = f"{sub_ses} session {session}"

    if subj_dir.exists():
        msg = (
            f"Recon for {sub_ses} is already present. Please delete the "
            f"directory if you want to recompute."
        )
        logger.info(**gen_log_kwargs(message=msg))
        return
    msg = (
        "Running recon-all on {sub_ses}. This will take "
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
        f"{root_dir}",
        f"{subjects_dir}",
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


def _has_session_specific_anat(subject, session, subjects_dir):
    return (subjects_dir / f"sub-{subject}" / f"ses-{session}").exists()


def main(*, config) -> None:
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
    root_dir = config.bids_root
    subjects_dir = Path(get_fs_subjects_dir(config))
    subjects_dir.mkdir(parents=True, exist_ok=True)

    # check for session-specific MRIs within subject, and handle accordingly
    subj_sess = list()
    for _subj in subjects:
        for _sess in sessions:
            session = (
                _sess
                if _has_session_specific_anat(_subj, _sess, subjects_dir)
                else None
            )
            subj_sess.append((_subj, session))

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(run_recon, exec_params=config.exec_params)
        parallel(
            run_func(root_dir, subject, fs_bids_app, subjects_dir, session)
            for subject, session in subj_sess
        )

        # Handle fsaverage
        fsaverage_dir = subjects_dir / "fsaverage"
        if fsaverage_dir.exists():
            if fsaverage_dir.is_symlink():
                fsaverage_dir.unlink()
            else:
                shutil.rmtree(fsaverage_dir)

        env = os.environ
        shutil.copytree(
            f"{env['FREESURFER_HOME']}/subjects/fsaverage", subjects_dir / "fsaverage"
        )
