#!/usr/bin/env python

from pathlib import Path
import logging
from typing import Union
from types import SimpleNamespace

import mne.bem

import config
from config import parallel_func, on_error, failsafe_run

PathLike = Union[str, Path]
logger = logging.getLogger('mne-bids-pipeline')
fs_bids_app = Path(__file__).parent / 'contrib' / 'run.py'


def make_coreg_surfaces(
    cfg: SimpleNamespace,
    subject: str
) -> None:
    """Create head surfaces for use with MNE-Python coregistration tools."""
    fs_subject = config.get_fs_subject(subject)
    subject_str = f'sub-{subject}' if subject != 'fsaverage' else 'fsaverage'
    logger.info(
        f'Creating scalp surfaces for coregistration, '
        f'subject: {subject_str} (FreeSurfer subject: {fs_subject})'
    )

    mne.bem.make_scalp_surfaces(
        subject=fs_subject,
        subjects_dir=cfg.subjects_dir,
        force=True,
        overwrite=True
    )


def get_config() -> SimpleNamespace:
    cfg = SimpleNamespace(
        subjects_dir=config.get_fs_subjects_dir()
    )
    return cfg


@failsafe_run(on_error=on_error, script_path=__file__)
def main():
    # Ensure we're also processing fsaverage if present
    subjects = config.get_subjects()
    if (Path(config.get_fs_subjects_dir()) / 'fsaverage').exists():
        subjects.append('fsaverage')

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(make_coreg_surfaces)

        parallel(
            run_func(
                get_config(), subject
            ) for subject in subjects
        )


if __name__ == '__main__':
    main()
