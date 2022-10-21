#!/usr/bin/env python
"""Generate coregistration surfaces.

Use FreeSurfer's ``mkheadsurf`` and related utilities to make head surfaces
suitable for coregistration.
"""
from pathlib import Path
import logging
from typing import Union
from types import SimpleNamespace

import mne.bem

from ..._config_utils import (
    get_fs_subjects_dir, get_fs_subject, get_subjects, _get_scalp_in_files,
)
from ..._logging import gen_log_kwargs
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run

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


def get_output_fnames_coreg_surfaces(*, cfg, subject):
    out_files = dict()
    subject_path = Path(cfg.subjects_dir) / cfg.fs_subject
    out_files['seghead'] = subject_path / 'surf' / 'lh.seghead'
    for key in ('dense', 'medium', 'sparse'):
        out_files[f'head-{key}'] = \
            subject_path / 'bem' / f'{cfg.fs_subject}-head-{key}.fif'
    return out_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_coreg_surfaces,
              get_output_fnames=get_output_fnames_coreg_surfaces)
def make_coreg_surfaces(
    cfg: SimpleNamespace,
    subject: str,
    in_files: dict,
) -> dict:
    """Create head surfaces for use with MNE-Python coregistration tools."""
    msg = 'Creating scalp surfaces for coregistration'
    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    in_files.pop('t1' if 't1' in in_files else 'seghead')
    mne.bem.make_scalp_surfaces(
        subject=cfg.fs_subject,
        subjects_dir=cfg.subjects_dir,
        force=True,
        overwrite=True
    )
    out_files = get_output_fnames_coreg_surfaces(cfg=cfg, subject=subject)
    return out_files


def get_config(*, config, subject) -> SimpleNamespace:
    cfg = SimpleNamespace(
        subject=subject,
        fs_subject=get_fs_subject(config, subject),
        subjects_dir=get_fs_subjects_dir(config),
    )
    return cfg


def main():
    import config
    # Ensure we're also processing fsaverage if present
    subjects = get_subjects(config)
    if (Path(get_fs_subjects_dir(config)) / 'fsaverage').exists():
        subjects.append('fsaverage')

    with get_parallel_backend(config):
        parallel, run_func = parallel_func(make_coreg_surfaces, config=config)

        parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                subject=subject,
                force_run=config.recreate_scalp_surface,
            ) for subject in subjects
        )


if __name__ == '__main__':
    main()
