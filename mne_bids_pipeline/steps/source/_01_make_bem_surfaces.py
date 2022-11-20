"""Create BEM surfaces.

Generate the BEM surfaces from a T1 or FLASH MRI scan.
"""

import glob
from pathlib import Path
from types import SimpleNamespace

import mne

from ..._config_utils import (
    get_fs_subject, get_subjects, _get_bem_conductivity, get_fs_subjects_dir)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import get_parallel_backend, parallel_func
from ..._run import failsafe_run, save_logs


def _get_bem_params(cfg: SimpleNamespace):
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


def get_input_fnames_make_bem_surfaces(
    *,
    cfg: SimpleNamespace,
    subject: str,
) -> dict:
    in_files = dict()
    mri_images, mri_dir, flash_dir = _get_bem_params(cfg)
    in_files['t1'] = mri_dir / 'T1.mgz'
    if mri_images == 'FLASH':
        flash_fnames = sorted(glob.glob(str(flash_dir / "mef*_*.mgz")))
        # We could check for existence here, but make_flash_bem does it later
        for fname in flash_fnames:
            in_files[fname.stem] = fname
    return in_files


def get_output_fnames_make_bem_surfaces(
    *,
    cfg: SimpleNamespace,
    subject: str,
) -> dict:
    out_files = dict()
    conductivity, _ = _get_bem_conductivity(cfg)
    n_layers = len(conductivity)
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    for surf in ('inner_skull', 'outer_skull', 'outer_skin')[:n_layers]:
        out_files[surf] = bem_dir / f'{surf}.surf'
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_make_bem_surfaces,
    get_output_fnames=get_output_fnames_make_bem_surfaces,
)
def make_bem_surfaces(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    in_files: dict,
) -> dict:
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
    show = True if exec_params.interactive else False
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
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=get_fs_subject(config=config, subject=subject),
        fs_subjects_dir=get_fs_subjects_dir(config=config),
        bem_mri_images=config.bem_mri_images,
        freesurfer_verbose=config.freesurfer_verbose,
        use_template_mri=config.use_template_mri,
        ch_types=config.ch_types,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
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
            mne.datasets.fetch_fsaverage(get_fs_subjects_dir(config))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            make_bem_surfaces, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
                force_run=config.recreate_bem)
            for subject in get_subjects(config)
        )
    save_logs(config=config, logs=logs)
