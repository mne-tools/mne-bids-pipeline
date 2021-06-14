"""
=================================
13. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import itertools
import logging

import mne
from mne.utils._bunch import BunchConst
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import (gen_log_message, on_error, failsafe_run, get_fs_subject,
                    sanitize_cond_name)

logger = logging.getLogger('mne-bids-pipeline')


def morph_stc(cfg, subject, session=None):
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

    fs_subject = get_fs_subject(subject)

    morphed_stcs = []

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
        stc_fsaverage.save(fname_stc_fsaverage)
        morphed_stcs.append(stc_fsaverage)

        del fname_stc, fname_stc_fsaverage

    return morphed_stcs


def get_config(subject, session):
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        subjects=config.get_subjects(),
        sessions=config.get_sessions(),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        deriv_root=config.get_deriv_root(),
        N_JOBS=config.N_JOBS,
        run_source_estimation=config.run_source_estimation
    )
    return cfg


@failsafe_run(on_error=on_error)
def main():
    """Run group average in source space"""
    msg = 'Running Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))

    cfg = get_config(None, None)

    if not cfg.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=13, message=msg))
        return

    mne.datasets.fetch_fsaverage(subjects_dir=cfg.fs_subjects_dir)

    parallel, run_func, _ = parallel_func(morph_stc, n_jobs=cfg.N_JOBS)
    all_morphed_stcs = parallel(
        run_func(get_config(subject, session), subject, session)
        for subject, session in
        itertools.product(cfg.subjects,
                          cfg.sessions)
    )
    all_morphed_stcs = [morphed_stcs for morphed_stcs, subject in
                        zip(all_morphed_stcs, cfg.subjects)]
    mean_morphed_stcs = map(sum, zip(*all_morphed_stcs))

    subject = 'average'
    # XXX to fix
    if cfg.sessions:
        session = cfg.sessions[0]
    else:
        session = None

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

    for condition, this_stc in zip(conditions, mean_morphed_stcs):
        this_stc /= len(all_morphed_stcs)

        method = cfg.inverse_method
        cond_str = sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph2fsaverage'

        fname_stc_avg = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}')
        this_stc.save(fname_stc_avg)

    msg = 'Completed Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))


if __name__ == '__main__':
    main()
