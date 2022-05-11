"""
===============================
11. Noise covariance estimation
===============================

Covariance matrices are computed and saved.
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

import config
from config import (
    gen_log_kwargs, on_error, failsafe_run, parallel_func,
    get_noise_cov_bids_path
)

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
    cov_fname = get_noise_cov_bids_path(
        noise_cov=config.noise_cov,
        cfg=cfg,
        subject=subject,
        session=session
    )

    msg = (f"Computing regularized covariance based on epochs' baseline "
           f"periods. Input: {epo_fname.basename}, "
           f"Output: {cov_fname.basename}")
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(epo_fname, preload=True)
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk',
                                 rank='info')
    cov.save(cov_fname, overwrite=True)


def compute_cov_from_raw(cfg, subject, session):
    bids_path_raw_noise = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        processing='filt',
        suffix='raw',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    data_type = ('resting-state' if config.noise_cov == 'rest' else
                 'empty-room')

    if data_type == 'resting-state':
        bids_path_raw_noise.task = 'rest'
    else:
        bids_path_raw_noise.task = 'noise'

    cov_fname = get_noise_cov_bids_path(
        noise_cov=config.noise_cov,
        cfg=cfg,
        subject=subject,
        session=session
    )

    msg = (f'Computing regularized covariance based on {data_type} recording. '
           f'Input: {bids_path_raw_noise.basename}, '
           f'Output: {cov_fname.basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    raw_noise = mne.io.read_raw_fif(bids_path_raw_noise, preload=True)
    cov = mne.compute_raw_covariance(raw_noise, method='shrunk', rank='info')
    cov.save(cov_fname, overwrite=True)


def retrieve_custom_cov(
    cfg: SimpleNamespace,
    subject: str,
    session: str
):
    assert callable(config.noise_cov)

    evoked_bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        processing=cfg.proc,
        recording=cfg.rec,
        space=cfg.space,
        suffix='ave',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )
    cov_fname = get_noise_cov_bids_path(
        noise_cov=config.noise_cov,
        cfg=cfg,
        subject=subject,
        session=session
    )

    msg = (f'Retrieving noise covariance matrix from custom user-supplied '
           f'function, Output: {cov_fname.basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    cov = config.noise_cov(evoked_bids_path)
    assert isinstance(cov, mne.Covariance)
    cov.save(cov_fname, overwrite=True)


@failsafe_run(on_error=on_error, script_path=__file__)
def run_covariance(*, cfg, subject, session=None, custom_func=None):
    if callable(config.noise_cov):
        retrieve_custom_cov(
            cfg=cfg, subject=subject, session=session
        )
    elif (
        (config.noise_cov == 'emptyroom' and 'eeg' not in cfg.ch_types) or
        config.noise_cov == 'rest'
    ):
        compute_cov_from_raw(cfg=cfg, subject=subject, session=session)
    else:
        tmin, tmax = config.noise_cov
        compute_cov_from_epochs(cfg=cfg, subject=subject, session=session,
                                tmin=tmin, tmax=tmax)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        spatial_filter=config.spatial_filter,
        ch_types=config.ch_types,
        deriv_root=config.get_deriv_root(),
        run_source_estimation=config.run_source_estimation,
    )
    return cfg


def main():
    """Run cov."""
    cfg = get_config()

    if not cfg.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    # Note that we're using config.noise_cov here and not adding it to
    # cfg, as in case it's a function, it won't work when running parallel jobs

    if config.noise_cov == "ad-hoc":
        msg = '    … skipping: using ad-hoc diagonal covariance.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_covariance)
        logs = parallel(
            run_func(cfg=cfg, subject=subject, session=session)
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )
        config.save_logs(logs)


if __name__ == '__main__':
    main()
