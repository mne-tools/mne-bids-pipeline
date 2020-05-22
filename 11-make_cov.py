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


@failsafe_run(on_error=on_error)
def run_covariance(subject, session=None):
    # Construct the search path for the data file. `sub` is mandatory
    kind = config.get_kind()
    subject_path = config.get_subject_path(subject=subject, session=session,
                                           kind=kind)
    deriv_path = op.join(config.deriv_root, subject_path)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    if config.use_ica or config.use_ssp:
        extension = '_cleaned-epo'
    else:
        extension = '-epo'

    fname_epo = op.join(deriv_path, bids_basename + '%s.fif' % extension)
    fname_cov = op.join(deriv_path, bids_basename + '-cov.fif')

    msg = f'Input: {fname_epo}, Output: {fname_cov}'
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    epochs = mne.read_epochs(fname_epo, preload=True)

    msg = 'Computing regularized covariance'
    logger.info(gen_log_message(message=msg, step=11, subject=subject,
                                session=session))

    # Do not shuffle the data before splitting into train and test samples.
    # Perform a block cross-validation instead to maintain autocorrelated
    # noise.
    cv = KFold(3, shuffle=False)
    cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', cv=cv,
                                 rank='info')
    cov.save(fname_cov)


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
