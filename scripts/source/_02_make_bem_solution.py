"""
====================
Compute BEM solution
====================

Compute the BEM solution.
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


def get_input_fnames_make_bem_solution(*, cfg, subject):
    in_files = dict()
    conductivity, _ = _get_bem_conductivity(cfg)
    n_layers = len(conductivity)
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    for surf in ('inner_skull', 'outer_skull', 'outer_skin')[:n_layers]:
        in_files[surf] = bem_dir / f'{surf}.surf'
    return in_files


def get_output_fnames_make_bem_solution(*, cfg, subject):
    out_files = dict()
    bem_dir = Path(cfg.fs_subjects_dir) / cfg.fs_subject / 'bem'
    _, tag = _get_bem_conductivity(cfg)
    out_files['model'] = bem_dir / f'{cfg.fs_subject}-{tag}-bem.fif'
    out_files['sol'] = bem_dir / f'{cfg.fs_subject}-{tag}-bem-sol.fif'
    return out_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_make_bem_solution,
              get_output_fnames=get_output_fnames_make_bem_solution,
              force_run=config.recreate_bem)
def make_bem_solution(*, cfg, subject, in_files):
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
    subject: Optional[str] = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        ch_types=config.ch_types,
        use_template_mri=config.use_template_mri,
    )
    return cfg


def main():
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
            mne.datasets.fetch_fsaverage(config.get_fs_subjects_dir())
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(make_bem_solution)
        logs = parallel(
            run_func(cfg=get_config(subject=subject), subject=subject)
            for subject in config.get_subjects()
        )
        config.save_logs(logs)


if __name__ == '__main__':
    main()
