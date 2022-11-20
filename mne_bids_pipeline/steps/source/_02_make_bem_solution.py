"""Compute BEM solution.

Compute the BEM solution.
"""

from pathlib import Path
from types import SimpleNamespace

import mne

from ..._config_utils import (
    _get_bem_conductivity, get_fs_subjects_dir, get_fs_subject, get_subjects)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run, save_logs


def get_input_fnames_make_bem_solution(
    *,
    cfg: SimpleNamespace,
    subject: str,
) -> dict:
    in_files = dict()
    conductivity, _ = _get_bem_conductivity(cfg)
    n_layers = len(conductivity)
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    for surf in ('inner_skull', 'outer_skull', 'outer_skin')[:n_layers]:
        in_files[surf] = bem_dir / f'{surf}.surf'
    return in_files


def get_output_fnames_make_bem_solution(
    *,
    cfg: SimpleNamespace,
    subject: str,
) -> dict:
    out_files = dict()
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    _, tag = _get_bem_conductivity(cfg)
    out_files['model'] = bem_dir / f'{cfg.fs_subject}-{tag}-bem.fif'
    out_files['sol'] = bem_dir / f'{cfg.fs_subject}-{tag}-bem-sol.fif'
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_make_bem_solution,
    get_output_fnames=get_output_fnames_make_bem_solution,
)
def make_bem_solution(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    in_files: dict,
) -> dict:
    msg = 'Calculating BEM solution'
    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    conductivity, _ = _get_bem_conductivity(cfg)
    bem_model = mne.make_bem_model(subject=cfg.fs_subject,
                                   subjects_dir=cfg.fs_subjects_dir,
                                   conductivity=conductivity)
    bem_sol = mne.make_bem_solution(bem_model)
    out_files = get_output_fnames_make_bem_solution(cfg=cfg, subject=subject)
    mne.write_bem_surfaces(out_files['model'], bem_model, overwrite=True)
    mne.write_bem_solution(out_files['sol'], bem_sol, overwrite=True)
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=get_fs_subject(config=config, subject=subject),
        fs_subjects_dir=get_fs_subjects_dir(config),
        ch_types=config.ch_types,
        use_template_mri=config.use_template_mri,
    )
    return cfg


def main(*, config) -> None:
    """Run BEM solution calculation."""
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    if config.use_template_mri is not None:
        msg = ('Skipping, BEM solution computation not needed for '
               'MRI template …')
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        if config.use_template_mri == "fsaverage":
            # Ensure we have the BEM
            mne.datasets.fetch_fsaverage(get_fs_subjects_dir(config))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            make_bem_solution, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                force_run=config.recreate_bem)
            for subject in get_subjects(config)
        )
    save_logs(config=config, logs=logs)
