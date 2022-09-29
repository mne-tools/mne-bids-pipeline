#!/usr/bin/env python

from pathlib import Path
import logging
from typing import Union
from types import SimpleNamespace

import mne.bem

import config
from config import parallel_func, failsafe_run, _get_scalp_in_files

PathLike = Union[str, Path]
logger = logging.getLogger('mne-bids-pipeline')
fs_bids_app = Path(__file__).parent / 'contrib' / 'run.py'


def get_input_fnames_coreg_surfaces(**kwargs):
    cfg = kwargs.pop('cfg')
    kwargs.pop('subject')  # unused
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    in_files = _get_scalp_in_files(cfg)
    return in_files


# TODO: Maybe deduplicate with _01_make_bem_surfaces.py?
@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_coreg_surfaces)
def make_coreg_surfaces(
    cfg: SimpleNamespace,
    subject: str,
    in_files: dict,
) -> dict:
    """Create head surfaces for use with MNE-Python coregistration tools."""
    subject_str = f'sub-{subject}' if subject != 'fsaverage' else 'fsaverage'
    subject_path = Path(cfg.subjects_dir) / cfg.fs_subject
    logger.info(
        f'Creating scalp surfaces for coregistration, '
        f'subject: {subject_str} (FreeSurfer subject: {cfg.fs_subject})'
    )
    in_files.pop('t1' if 't1' in in_files else 'seghead')
    mne.bem.make_scalp_surfaces(
        subject=cfg.fs_subject,
        subjects_dir=cfg.subjects_dir,
        force=True,
        overwrite=True
    )
    out_files = dict()
    out_files['seghead'] = subject_path / 'surf' / 'lh.seghead'
    for key in ('dense', 'medium', 'sparse'):
        out_files[f'head-{key}'] = \
            subject_path / 'bem' / f'{cfg.fs_subject}-head-{key}.fif'
    return out_files


def get_config(*, subject) -> SimpleNamespace:
    cfg = SimpleNamespace(
        subject=subject,
        fs_subject=config.get_fs_subject(subject),
        subjects_dir=config.get_fs_subjects_dir(),
    )
    return cfg


@failsafe_run(script_path=__file__)
def main():
    # Ensure we're also processing fsaverage if present
    subjects = config.get_subjects()
    if (Path(config.get_fs_subjects_dir()) / 'fsaverage').exists():
        subjects.append('fsaverage')

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(make_coreg_surfaces)

        parallel(
            run_func(
                cfg=get_config(subject=subject),
                subject=subject
            ) for subject in subjects
        )


if __name__ == '__main__':
    main()
