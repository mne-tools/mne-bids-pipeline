"""
=================================
13. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import itertools
import logging
from typing import Optional

import numpy as np
import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run, sanitize_cond_name

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
            stc, subject_from=cfg.fs_subject, subject_to='fsaverage',
            subjects_dir=cfg.fs_subjects_dir)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(fname_stc_fsaverage)
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
        stc.save(fname_stc_avg)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        deriv_root=config.get_deriv_root(),
    )
    return cfg


@failsafe_run(on_error=on_error)
def main():
    """Run group average in source space"""
    msg = 'Running Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))

    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=13, message=msg))
        return

    mne.datasets.fetch_fsaverage(subjects_dir=config.get_fs_subjects_dir())

    parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)
    all_morphed_stcs = parallel(
        run_func(get_config(subject, session), subject, session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )
    mean_morphed_stcs = np.array(all_morphed_stcs).mean(axis=0)

    # XXX to fix
    sessions = config.get_sessions()
    if sessions:
        session = sessions[0]
    else:
        session = None

    run_average(
        cfg=get_config(session=session),
        session=session,
        mean_morphed_stcs=mean_morphed_stcs
    )

    msg = 'Completed Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))


if __name__ == '__main__':
    main()
