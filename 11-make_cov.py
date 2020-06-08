"""
==================================
08. Baseline covariance estimation
==================================

Covariance matrices are computed and saved.
"""

import os.path as op
import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

from sklearn.model_selection import KFold

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def compute_cov_from_epochs(subject, session, tmin, tmax):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    if config.use_ica or config.use_ssp:
        extension = '_cleaned-epo'
    else:
        extension = '-epo'

    epo_fname = op.join(deriv_path, bids_basename + '%s.fif' % extension)
    cov_fname = op.join(deriv_path, bids_basename + '-cov.fif')

    msg = (f"Computing regularized covariance based on epochs' baseline "
           f"periods. Input: {epo_fname}, Output: {cov_fname}")
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    epochs = mne.read_epochs(epo_fname, preload=True)
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk',
                                 rank='info')
    cov.save(cov_fname)


def compute_cov_from_empty_room(subject, session):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    raw_er_fname = op.join(deriv_path,
                           bids_basename + '_emptyroom_filt_raw.fif')
    cov_fname = op.join(deriv_path, bids_basename + '-cov.fif')

    extra_params = dict()
    if not config.use_maxwell_filter and config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    msg = (f'Computing regularized covariance based on empty-room recording. '
           f'Input: {raw_er_fname}, Output: {cov_fname}')
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    raw_er = mne.io.read_raw_fif(raw_er_fname, preload=True, **extra_params)
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

    parallel, run_func, _ = parallel_func(run_covariance, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 11: Estimate noise covariance'
    logger.info(gen_log_message(step=11, message=msg))


if __name__ == '__main__':
    main()
