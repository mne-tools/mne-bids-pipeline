"""
===================
Create BEM surfaces
===================

Generate the BEM surfaces from a T1 or FLASH MRI scan. This is required to
produce the conductivity model and forward solution in the next step.

This script will also create a high-resolution surface of the scalp, which can
be used for visualization of the sensor alignment (coregistration).
"""

import logging
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import mne

import config
from config import gen_log_kwargs, failsafe_run
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


def make_bem(*, cfg, subject):
    fs_subject = cfg.fs_subject
    mri_dir = Path(cfg.fs_subjects_dir) / fs_subject / 'mri'
    bem_dir = Path(cfg.fs_subjects_dir) / fs_subject / 'bem'
    watershed_bem_dir = bem_dir / 'watershed'
    flash_bem_dir = bem_dir / 'flash'
    flash_dir = mri_dir / 'flash' / 'parameter_maps'
    show = True if cfg.interactive else False

    if cfg.bem_mri_images == 'FLASH' and not flash_dir.exists():
        raise RuntimeError('Cannot locate FLASH MRI images.')
    elif cfg.bem_mri_images == 'FLASH':
        mri_images = 'FLASH'
    elif cfg.bem_mri_images == 'auto' and flash_dir.exists():
        mri_images = 'FLASH'
    else:
        mri_images = 'T1'

    if ((mri_images == 'FLASH' and flash_bem_dir.exists()) or
            (mri_images == 'T1' and watershed_bem_dir.exists())):
        msg = 'Found existing BEM surfaces. '
        if cfg.recreate_bem:
            msg += 'Overwriting as requested in configuration.'
            logger.info(**gen_log_kwargs(message=msg, subject=subject))
        else:
            msg = 'Skipping surface extraction as requested in configuration.'
            logger.info(**gen_log_kwargs(message=msg, subject=subject))
            return

    if mri_images == 'FLASH':
        msg = 'Creating BEM surfaces from FLASH MRI images'
        bem_func = mne.bem.make_flash_bem
    else:
        msg = ('Creating BEM surfaces from T1-weighted MRI images using '
               'watershed algorithm')
        bem_func = mne.bem.make_watershed_bem

    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    bem_func(
        subject=fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        copy=True,
        overwrite=True,
        show=show,
        verbose=cfg.freesurfer_verbose
    )


def make_scalp_surface(*, cfg, subject):
    fs_subject = cfg.fs_subject
    bem_dir = Path(cfg.fs_subjects_dir) / fs_subject / 'bem'

    generate_surface = cfg.recreate_scalp_surface

    # Even if the user didn't ask for re-creation of the surface, we check if
    # the required file exist; if it doesn't, we create it
    surface_fname = bem_dir / f'sub-{subject}-head-dense.fif'
    if not generate_surface and not surface_fname.exists():
        generate_surface = True

    if not generate_surface:
        # Seems everything is in place, so we can safely skip surface creation
        msg = 'Not generating high-resolution scalp surface.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject))
        return

    msg = 'Generating high-resolution scalp surface for coregistration.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject))

    mne.bem.make_scalp_surfaces(
        subject=fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        no_decimate=True,
        force=True,
        overwrite=True,
        verbose=cfg.freesurfer_verbose
    )


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        recreate_bem=config.recreate_bem,
        bem_mri_images=config.bem_mri_images,
        recreate_scalp_surface=config.recreate_scalp_surface,
        interactive=config.interactive,
        freesurfer_verbose=config.freesurfer_verbose
    )
    return cfg


@failsafe_run(script_path=__file__)
def make_bem_and_scalp_surface(*, cfg, subject):
    make_bem(cfg=cfg, subject=subject)
    make_scalp_surface(cfg=cfg, subject=subject)


def main():
    """Run BEM surface extraction."""
    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    if config.use_template_mri is not None:
        msg = '    … skipping BEM computating when using MRI template.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(make_bem_and_scalp_surface)
        logs = parallel(
            run_func(cfg=get_config(subject=subject), subject=subject)
            for subject in config.get_subjects()
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
