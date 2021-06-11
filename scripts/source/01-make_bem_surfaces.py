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

import mne
from mne.parallel import parallel_func

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def make_bem(subject):
    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()
    mri_dir = Path(fs_subjects_dir) / fs_subject / 'mri'
    bem_dir = Path(fs_subjects_dir) / fs_subject / 'bem'
    watershed_bem_dir = bem_dir / 'watershed'
    flash_bem_dir = bem_dir / 'flash'
    flash_dir = mri_dir / 'flash' / 'parameter_maps'
    show = True if config.interactive else False

    if config.bem_mri_images == 'FLASH' and not flash_dir.exists():
        raise RuntimeError('Cannot locate FLASH MRI images.')
    elif config.bem_mri_images == 'FLASH':
        mri_images = 'FLASH'
    elif config.bem_mri_images == 'auto' and flash_dir.exists():
        mri_images = 'FLASH'
    else:
        mri_images = 'T1'

    if ((mri_images == 'FLASH' and flash_bem_dir.exists()) or
            (mri_images == 'T1' and watershed_bem_dir.exists())):
        msg = 'Found existing BEM surfaces. '
        if config.recreate_bem:
            msg += 'Overwriting as requested in configuration.'
            logger.info(gen_log_message(step=10, message=msg, subject=subject))
        else:
            msg = 'Skipping surface extraction as requested in configuration.'
            logger.info(gen_log_message(step=10, message=msg, subject=subject))
            return

    if mri_images == 'FLASH':
        msg = 'Creating BEM surfaces from FLASH MRI images'
        bem_func = mne.bem.make_flash_bem
    else:
        msg = ('Creating BEM surfaces from T1-weighted MRI images using '
               'watershed algorithm')
        bem_func = mne.bem.make_watershed_bem

    logger.info(gen_log_message(step=10, message=msg, subject=subject))
    bem_func(subject=fs_subject,
             subjects_dir=fs_subjects_dir,
             copy=True,
             overwrite=True,
             show=show)


@failsafe_run(on_error=on_error)
def make_scalp_surface(subject):
    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()
    bem_dir = Path(fs_subjects_dir) / fs_subject / 'bem'

    generate_surface = config.recreate_scalp_surface

    # Even if the user didn't ask for re-creation of the surface, we check if
    # the required file exist; if it doesn't, we create it
    surface_fname = bem_dir / f'sub-{subject}-head-dense.fif'
    if not generate_surface and not surface_fname.exists():
        generate_surface = True

    if not generate_surface:
        # Seems everything is in place, so we can safely skip surface creation
        msg = 'Not generating high-resolution scalp surface.'
        logger.info(gen_log_message(step=10, message=msg, subject=subject))
        return

    msg = 'Generating high-resolution scalp surface for coregistration.'
    logger.info(gen_log_message(step=10, message=msg, subject=subject))

    mne.bem.make_scalp_surfaces(
        subject=fs_subject,
        subjects_dir=fs_subjects_dir,
        no_decimate=True,
        force=True,
        overwrite=True
    )


def main():
    """Run BEM surface extraction."""
    msg = 'Running Step 10: Create BEM & high-resolution scalp surface'
    logger.info(gen_log_message(step=10, message=msg))

    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=10, message=msg))
        return

    parallel, run_func, _ = parallel_func(make_bem, n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.get_subjects())

    parallel, run_func, _ = parallel_func(make_scalp_surface,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.get_subjects())

    msg = 'Completed Step 10: Create BEM & high-resolution scalp surface'
    logger.info(gen_log_message(step=10, message=msg))


if __name__ == '__main__':
    main()
