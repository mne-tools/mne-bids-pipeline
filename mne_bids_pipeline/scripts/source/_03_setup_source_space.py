"""Setup source space.

Set up source space for forward and inverse computation.
"""

import logging
from typing import Optional
from types import SimpleNamespace

import mne

import config
from config import gen_log_kwargs, failsafe_run
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


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


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_setup_source_space,
              get_output_fnames=get_output_fnames_setup_source_space,
              force_run=False)  # should never need to force run
def run_setup_source_space(*, cfg, subject, in_files):
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
    subject: Optional[str] = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        spacing=config.spacing,
        use_template_mri=config.use_template_mri,
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
    )
    return cfg


def main():
    """Run forward."""
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    if config.use_template_mri is not None:
        subjects = [config.use_template_mri]
    else:
        subjects = config.get_subjects()

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_setup_source_space)
        logs = parallel(
            run_func(
                cfg=get_config(subject=subject), subject=subject,
            )
            for subject in subjects
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
