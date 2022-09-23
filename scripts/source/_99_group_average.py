"""
=================================
13. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import itertools
import logging
from types import SimpleNamespace

import numpy as np

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, failsafe_run, sanitize_cond_name
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


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


def run_average(cfg, session, mean_morphed_stcs):
    subject = 'average'
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


def get_config() -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        task_is_rest=config.task_is_rest,
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        fs_subjects_dir=config.get_fs_subjects_dir(),
        deriv_root=config.get_deriv_root(),
    )
    return cfg


# pass 'average' subject for logging
@failsafe_run(script_path=__file__)
def run_group_average_source(*, cfg, subject='average'):
    """Run group average in source space"""

    mne.datasets.fetch_fsaverage(subjects_dir=config.get_fs_subjects_dir())

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(morph_stc)
        all_morphed_stcs = parallel(
            run_func(
                cfg=cfg, subject=subject,
                fs_subject=config.get_fs_subject(subject),
                session=session
            )
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )
        mean_morphed_stcs = np.array(all_morphed_stcs).mean(axis=0)

        # XXX to fix
        sessions = config.get_sessions()
        if sessions:
            session = sessions[0]
        else:
            session = None

        run_average(
            cfg=cfg,
            session=session,
            mean_morphed_stcs=mean_morphed_stcs
        )


def main():
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    log = run_group_average_source(cfg=get_config())
    config.save_logs([log])


if __name__ == '__main__':
    main()
