"""Setup source space.

Set up source space for forward and inverse computation.
"""

from types import SimpleNamespace

import mne

from ..._config_utils import (
    get_fs_subject, get_fs_subjects_dir, get_subjects)
from ..._logging import logger, gen_log_kwargs
from ..._run import failsafe_run, save_logs
from ..._parallel import parallel_func, get_parallel_backend


def get_input_fnames_setup_source_space(*, cfg, subject):
    in_files = dict()
    surf_path = cfg.fs_subjects_dir / cfg.fs_subject / 'surf'
    for hemi in ('lh', 'rh'):
        for kind in ('sphere', 'sphere.reg', 'white'):
            in_files['surf-{hemi}-{kind}'] = surf_path / f'{hemi}.{kind}'
    return in_files


def get_output_fnames_setup_source_space(*, cfg, subject):
    out_files = dict()
    out_files['src'] = (cfg.fs_subjects_dir / cfg.fs_subject / 'bem' /
                        f'{cfg.fs_subject}-{cfg.spacing}-src.fif')
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_setup_source_space,
    get_output_fnames=get_output_fnames_setup_source_space,
)
def run_setup_source_space(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    in_files: dict,
) -> dict:
    msg = f'Creating source space with spacing {repr(cfg.spacing)}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    src = mne.setup_source_space(
        cfg.fs_subject, spacing=cfg.spacing, subjects_dir=cfg.fs_subjects_dir,
        add_dist='patch')
    in_files.clear()  # all used by setup_source_space
    out_files = get_output_fnames_setup_source_space(cfg=cfg, subject=subject)
    mne.write_source_spaces(out_files['src'], src, overwrite=True)
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        spacing=config.spacing,
        use_template_mri=config.use_template_mri,
        fs_subject=get_fs_subject(config=config, subject=subject),
        fs_subjects_dir=get_fs_subjects_dir(config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run forward."""
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    if config.use_template_mri is not None:
        subjects = [
            config.use_template_mri
        ]
    else:
        subjects = get_subjects(config=config)

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_setup_source_space, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
            )
            for subject in subjects
        )
    save_logs(config=config, logs=logs)
