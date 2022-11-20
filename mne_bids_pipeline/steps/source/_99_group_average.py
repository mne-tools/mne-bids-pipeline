"""Group average at the source level.

Source estimates are morphed to the ``fsaverage`` brain.
"""

from types import SimpleNamespace
from typing import Optional, List

import numpy as np

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_fs_subjects_dir, get_subjects, sanitize_cond_name, get_fs_subject,
    get_task, get_datatype, get_sessions, get_all_contrasts,
)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import run_report_average_source
from ..._run import failsafe_run, save_logs


def morph_stc(cfg, subject, fs_subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    morphed_stcs = []

    if cfg.task_is_rest:
        conditions = [cfg.task.lower()]
    else:
        if isinstance(cfg.conditions, dict):
            conditions = list(cfg.conditions.keys())
        else:
            conditions = cfg.conditions

    for condition in conditions:
        method = cfg.inverse_method
        cond_str = sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph2fsaverage'

        fname_stc = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{hemi_str}')
        fname_stc_fsaverage = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}')

        stc = mne.read_source_estimate(fname_stc)

        morph = mne.compute_source_morph(
            stc, subject_from=fs_subject, subject_to='fsaverage',
            subjects_dir=cfg.fs_subjects_dir)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(fname_stc_fsaverage, overwrite=True)
        morphed_stcs.append(stc_fsaverage)

        del fname_stc, fname_stc_fsaverage

    return morphed_stcs


def run_average(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    mean_morphed_stcs: List[mne.SourceEstimate],
):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         processing=cfg.proc,
                         recording=cfg.rec,
                         space=cfg.space,
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions

    for condition, stc in zip(conditions, mean_morphed_stcs):
        method = cfg.inverse_method
        cond_str = sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph2fsaverage'

        fname_stc_avg = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}')
        stc.save(fname_stc_avg, overwrite=True)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        task_is_rest=config.task_is_rest,
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        fs_subjects_dir=get_fs_subjects_dir(config),
        deriv_root=config.deriv_root,
        subjects_dir=get_fs_subjects_dir(config),
        bids_root=config.bids_root,
        data_type=config.data_type,
        ch_types=config.ch_types,
        subjects=config.subjects,
        exclude_subjects=config.exclude_subjects,
        sessions=get_sessions(config),
        use_template_mri=config.use_template_mri,
        all_contrasts=get_all_contrasts(config),
        report_stc_n_time_points=config.report_stc_n_time_points,
    )
    return cfg


# pass 'average' subject for logging
@failsafe_run()
def run_group_average_source(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
) -> None:
    """Run group average in source space"""

    mne.datasets.fetch_fsaverage(subjects_dir=get_fs_subjects_dir(cfg))

    with get_parallel_backend(exec_params):
        parallel, run_func = parallel_func(
            morph_stc, exec_params=exec_params)
        all_morphed_stcs = parallel(
            run_func(
                cfg=cfg, subject=subject,
                fs_subject=get_fs_subject(config=cfg, subject=subject),
                session=session
            )
            for subject in get_subjects(cfg)
            for session in get_sessions(cfg)
        )
        mean_morphed_stcs = np.array(all_morphed_stcs).mean(axis=0)

        # XXX to fix
        sessions = get_sessions(cfg)
        if sessions:
            session = sessions[0]
        else:
            session = None

        run_average(
            cfg=cfg,
            session=session,
            subject=subject,
            mean_morphed_stcs=mean_morphed_stcs
        )
        run_report_average_source(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session,
        )


def main(*, config: SimpleNamespace) -> None:
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    log = run_group_average_source(
        cfg=get_config(
            config=config,
        ),
        exec_params=config.exec_params,
        subject='average',
    )
    save_logs(config=config, logs=[log])
