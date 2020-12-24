#!/usr/bin/env python

import os
import shutil
import sys
from pathlib import Path
import logging
from typing import Union

import fire

from mne.utils import run_subprocess
from mne.parallel import parallel_func

import config

PathLike = Union[str, Path]
logger = logging.getLogger('mne-study-template')
fs_bids_app = Path(__file__).parent / 'contrib' / 'run.py'


def _get_subjects_dir(root_dir) -> Path:
    subjects_dir = \
        Path(root_dir) / "derivatives" / "freesurfer" / "subjects"
    return subjects_dir


def run_recon(root_dir, subject, fs_bids_app, overwrite) -> None:
    logger.info(f"Running recon-all on subject {subject}.")

    subjects_dir = _get_subjects_dir(root_dir)
    subj_dir = subjects_dir / f"sub-{subject}"

    if subj_dir.exists():
        if overwrite is False:
            logger.info(f"Subject {subject} is already present. Set "
                        f"overwrite to True if want to recompute")
            return
        else:
            shutil.rmtree(subj_dir)

    env = os.environ

    if 'FREESURFER_HOME' not in env:
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
        f"{subjects_dir}", "participant",
        "--n_cpus=2", "--stages=all", "--skip_bids_validator",
        f"--license_file={license_file}",
        f"--participant_label={subject}"
    ]
    logger.debug("Running: " + " ".join(cmd))
    run_subprocess(cmd, env=env)


def main(overwrite: bool = False,
         n_jobs: int = 1) -> None:
    """Run freesurfer recon-all command on BIDS dataset.

    The command allows to run the freesurfer recon-all
    command on all subjects of your BIDS dataset. It can
    run in parallel with the --n_jobs parameter.

    It is built on top of the FreeSurfer BIDS app:

    https://github.com/BIDS-Apps/freesurfer

    You must have freesurfer available on your system.

    Examples
    --------
    run_freesurfer.py /path/to/bids/dataset/study-template-config.py /path/to/freesurfer_bids_app/

    or to run in parallel (3 subjects at a time):

    run_freesurfer.py /path/to/bids/dataset/study-template-config.py /path/to/freesurfer_bids_app/ --n_jobs=3

    """  # noqa

    logger.info('Running FreeSurfer')

    subjects = config.get_subjects()

    root_dir = config.bids_root
    subjects_dir = _get_subjects_dir(root_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    parallel, run_func, _ = parallel_func(run_recon, n_jobs=n_jobs)
    parallel(run_func(root_dir, subject, fs_bids_app, overwrite)
             for subject in subjects)

    # Handle fsaverage
    fsaverage_dir = subjects_dir / 'fsaverage'
    if fsaverage_dir.exists():
        if fsaverage_dir.is_symlink():
            fsaverage_dir.unlink()
        else:
            shutil.rmtree(fsaverage_dir)

    env = os.environ
    shutil.copytree(f"{env['FREESURFER_HOME']}/subjects/fsaverage",
                    subjects_dir / 'fsaverage')


if __name__ == '__main__':
    fire.Fire(main)
