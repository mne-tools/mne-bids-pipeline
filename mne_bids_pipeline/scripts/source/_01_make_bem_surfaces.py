"""Create BEM surfaces.

Generate the BEM surfaces from a T1 or FLASH MRI scan.
"""

import glob
import logging
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import mne

import config
from config import gen_log_kwargs, failsafe_run
from config import parallel_func, _get_bem_conductivity

logger = logging.getLogger('mne-bids-pipeline')


def _get_bem_params(cfg):
    mri_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'mri'
    flash_dir = mri_dir / 'flash' / 'parameter_maps'
    if cfg.bem_mri_images == 'FLASH' and not flash_dir.exists():
        raise RuntimeError('Cannot locate FLASH MRI images.')
    if cfg.bem_mri_images == 'FLASH':
        mri_images = 'FLASH'
    elif cfg.bem_mri_images == 'auto' and flash_dir.exists():
        mri_images = 'FLASH'
    else:
        mri_images = 'T1'
    return mri_images, mri_dir, flash_dir


def get_input_fnames_make_bem_surfaces(*, cfg, subject):
    in_files = dict()
    mri_images, mri_dir, flash_dir = _get_bem_params(cfg)
    in_files['t1'] = mri_dir / 'T1.mgz'
    if mri_images == 'FLASH':
        flash_fnames = sorted(glob.glob(str(flash_dir / "mef*_*.mgz")))
        # We could check for existence here, but make_flash_bem does it later
        for fname in flash_fnames:
            in_files[fname.stem] = fname
    return in_files


def get_output_fnames_make_bem_surfaces(*, cfg, subject):
    out_files = dict()
    conductivity, _ = _get_bem_conductivity(cfg)
    n_layers = len(conductivity)
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    for surf in ('inner_skull', 'outer_skull', 'outer_skin')[:n_layers]:
        out_files[surf] = bem_dir / f'{surf}.surf'
    return out_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_make_bem_surfaces,
              get_output_fnames=get_output_fnames_make_bem_surfaces,
              force_run=config.recreate_bem)
def make_bem_surfaces(*, cfg, subject, in_files):
    mri_images, _, _ = _get_bem_params(cfg)
    in_files.clear()  # assume we use everything we add
    if mri_images == 'FLASH':
        msg = 'Creating BEM surfaces from FLASH MRI images'
        bem_func = mne.bem.make_flash_bem
    else:
        msg = ('Creating BEM surfaces from T1-weighted MRI images using '
               'watershed algorithm')
        bem_func = mne.bem.make_watershed_bem
    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    show = True if cfg.interactive else False
    bem_func(
        subject=cfg.fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        copy=True,
        overwrite=True,
        show=show,
        verbose=cfg.freesurfer_verbose
    )
    out_files = get_output_fnames_make_bem_surfaces(cfg=cfg, subject=subject)
    return out_files


def get_config(
    subject: Optional[str] = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        bem_mri_images=config.bem_mri_images,
        interactive=config.interactive,
        freesurfer_verbose=config.freesurfer_verbose,
        use_template_mri=config.use_template_mri,
        ch_types=config.ch_types,
    )
    return cfg


def main():
    """Run BEM surface extraction."""
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    if config.use_template_mri is not None:
        msg = 'Skipping, BEM surface extraction not needed for MRI template …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        if config.use_template_mri == "fsaverage":
            # Ensure we have the BEM
            mne.datasets.fetch_fsaverage(config.get_fs_subjects_dir())
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(make_bem_surfaces)
        logs = parallel(
            run_func(cfg=get_config(subject=subject), subject=subject)
            for subject in config.get_subjects()
        )
        config.save_logs(logs)


if __name__ == '__main__':
    main()
