"""
===============================
11. Noise covariance estimation
===============================

Covariance matrices are computed and saved.
"""

import itertools
import logging
from typing import Optional

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


def compute_cov_from_epochs(cfg, subject, session, tmin, tmax):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         processing=cfg.proc,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    processing = None
    if cfg.spatial_filter is not None:
        processing = 'clean'

    epo_fname = bids_path.copy().update(processing=processing, suffix='epo')
    cov_fname = bids_path.copy().update(suffix='cov')

    msg = (f"Computing regularized covariance based on epochs' baseline "
           f"periods. Input: {epo_fname}, Output: {cov_fname}")
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(epo_fname, preload=True)
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk',
                                 rank='info')
    cov.save(cov_fname)


def compute_cov_from_empty_room(cfg, subject, session):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    raw_er_fname = bids_path.copy().update(processing='filt', task='noise',
                                           suffix='raw')
    cov_fname = bids_path.copy().update(suffix='cov')

    msg = (f'Computing regularized covariance based on empty-room recording. '
           f'Input: {raw_er_fname}, Output: {cov_fname}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    raw_er = mne.io.read_raw_fif(raw_er_fname, preload=True)
    cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank='info')
    cov.save(cov_fname)


@failsafe_run(on_error=on_error, script_path=__file__)
def run_covariance(*, cfg, subject, session=None):
    if cfg.noise_cov == 'emptyroom' and 'eeg' not in cfg.ch_types:
        compute_cov_from_empty_room(cfg=cfg, subject=subject, session=session)
    else:
        tmin, tmax = cfg.noise_cov
        compute_cov_from_epochs(cfg=cfg, subject=subject, session=session,
                                tmin=tmin, tmax=tmax)


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
        noise_cov=config.noise_cov,
        spatial_filter=config.spatial_filter,
        ch_types=config.ch_types,
        deriv_root=config.get_deriv_root(),
    )
    return cfg


def main():
    """Run cov."""
    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    if config.noise_cov == "ad-hoc":
        msg = '    … skipping: using ad-hoc diagonal covariance.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func, _ = parallel_func(run_covariance,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(), subject=subject, session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
