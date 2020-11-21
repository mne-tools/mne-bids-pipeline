#!/usr/bin/env python

import os
import shutil
import sys
from pathlib import Path
import logging
from typing import Union, Optional

import fire

from mne.utils import run_subprocess
from mne.parallel import parallel_func
from mne_bids import get_entity_vals

logger = logging.getLogger('mne-study-template')


def _get_subjects_dir(root_dir):
    subjects_dir = \
        Path(root_dir) / "derivatives" / "freesurfer" / "subjects"
    return subjects_dir


def run_recon(root_dir, fs_bids_app, subject, overwrite):
    logger.info(f"Running recon-all on subject {subject}.")

    subjects_dir = _get_subjects_dir(root_dir)

    if not subjects_dir.exists():
        subjects_dir.mkdir()

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
        raise RuntimeError("freesurfer is not available.")

    license_file = Path(f"{env['FREESURFER_HOME']}/license.txt")
    if not license_file.exists():
        license_file = Path(f"{env['FREESURFER_HOME']}/.license")
    if not license_file.exists():
        raise RuntimeError("freesurfer license file not found.")

    cmd = [
        f"{sys.executable}",
        f"{fs_bids_app}/run.py "
        f"{root_dir}",
        f"{subjects_dir} participant ",
        "--n_cpus=2 --stages=all --skip_bids_validator",
        f"--license_file={license_file}",
        f"--participant_label={subject}"
    ]
    logger.debug("Running: " + " ".join(cmd))
    run_subprocess(cmd, env=env, stdout=sys.stdout)


def main(root_dir: Union[str, Path],
         fs_bids_app: Union[str, Path],
         overwrite: Optional[bool] = False,
         n_jobs: Optional[int] = 1):

    logger.info('Running FreeSurfer')

    root_dir = str(Path(root_dir).expanduser())
    subjects = sorted(get_entity_vals(root_dir, entity_key='subject'))

    parallel, run_func, _ = parallel_func(run_recon, n_jobs=n_jobs)
    parallel(run_func(root_dir, fs_bids_app, subject,
                      overwrite) for subject in subjects)

    subjects_dir = _get_subjects_dir(root_dir)
    fsaverage_dir = subjects_dir / "fsaverage"

    if fsaverage_dir.exists():
        shutil.rmtree(fsaverage_dir)

    env = os.environ
    shutil.copytree(
        f"{env['FREESURFER_HOME']}/subjects/fsaverage",
        subjects_dir / 'fsaverage'
    )


if __name__ == '__main__':
    fire.Fire(main)
