"""
===============================
11. Noise covariance estimation
===============================

Covariance matrices are computed and saved.
"""

import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


def compute_cov_from_epochs(subject, session, tmin, tmax):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         processing=config.proc,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root(),
                         check=False)

    processing = None
    if config.spatial_filter is not None:
        processing = 'clean'

    epo_fname = bids_path.copy().update(processing=processing, suffix='epo')
    cov_fname = bids_path.copy().update(suffix='cov')

    msg = (f"Computing regularized covariance based on epochs' baseline "
           f"periods. Input: {epo_fname}, Output: {cov_fname}")
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    epochs = mne.read_epochs(epo_fname, preload=True)
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk',
                                 rank='info')
    cov.save(cov_fname)


def compute_cov_from_empty_room(subject, session):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root(),
                         check=False)

    raw_er_fname = bids_path.copy().update(processing='filt', task='noise',
                                           suffix='raw')
    cov_fname = bids_path.copy().update(suffix='cov')

    msg = (f'Computing regularized covariance based on empty-room recording. '
           f'Input: {raw_er_fname}, Output: {cov_fname}')
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    raw_er = mne.io.read_raw_fif(raw_er_fname, preload=True)
    cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank='info')
    cov.save(cov_fname)


@failsafe_run(on_error=on_error)
def run_covariance(subject, session=None):
    if config.noise_cov == 'emptyroom' and 'eeg' not in config.ch_types:
        compute_cov_from_empty_room(subject=subject, session=session)
    else:
        tmin, tmax = config.noise_cov
        compute_cov_from_epochs(subject=subject, session=session, tmin=tmin,
                                tmax=tmax)


def main():
    """Run cov."""
    msg = 'Running Step 11: Estimate noise covariance'
    logger.info(gen_log_message(step=11, message=msg))

    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=11, message=msg))
        return

    parallel, run_func, _ = parallel_func(run_covariance, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 11: Estimate noise covariance'
    logger.info(gen_log_message(step=11, message=msg))


if __name__ == '__main__':
    main()
