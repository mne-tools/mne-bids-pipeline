#!/usr/bin/env python

import os
import shutil
import sys
from pathlib import Path
import logging
from typing import Union

from mne.utils import run_subprocess
from config import parallel_func

import config

PathLike = Union[str, Path]
logger = logging.getLogger('mne-bids-pipeline')
fs_bids_app = Path(__file__).parent / 'contrib' / 'run.py'


def run_recon(root_dir, subject, fs_bids_app) -> None:
    subjects_dir = Path(config.get_fs_subjects_dir())
    subj_dir = subjects_dir / f"sub-{subject}"

    if subj_dir.exists():
        logger.info(f"Subject {subject} is already present. Please delete the "
                    f"directory if you want to recompute.")
        return
    logger.info(f"Running recon-all on subject {subject}. This will take "
                f"a LONG time – it's a good idea to let it run over night.")

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
    run_subprocess(cmd, env=env, verbose=logger.level)


def main() -> None:
    """Run freesurfer recon-all command on BIDS dataset.

    The script allows to run the freesurfer recon-all
    command on all subjects of your BIDS dataset. It can
    run in parallel with the --n_jobs parameter.

    It is built on top of the FreeSurfer BIDS app:

    https://github.com/BIDS-Apps/freesurfer

    and the MNE BIDS Pipeline

    https://mne.tools/mne-bids-pipeline

    You must have freesurfer available on your system.

    Run via the MNE BIDS Pipeline's `run.py`:

    python run.py --steps=freesurfer --config=your_pipeline_config.py

    """  # noqa

    logger.info('Running FreeSurfer')

    subjects = config.get_subjects()
    root_dir = config.get_bids_root()
    subjects_dir = Path(config.get_fs_subjects_dir())
    subjects_dir.mkdir(parents=True, exist_ok=True)

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_recon)
        parallel(run_func(root_dir, subject, fs_bids_app)
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
    main()
