"""
==================================
08. Baseline covariance estimation
==================================

Covariance matrices are computed and saved.
"""

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

from sklearn.model_selection import KFold
import numpy as np

import config


def run_covariance(subject, session=None):
    print("Processing subject: %s%s")
    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
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

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
    fname_epo = \
        op.join(fpath_deriv, bids_basename + '%s.fif' % extension)

    fname_cov = \
        op.join(fpath_deriv, bids_basename + '-cov.fif')

    print("Input: ", fname_epo)
    print("Output: ", fname_cov)

    epochs = mne.read_epochs(fname_epo, preload=True)

    print('  Computing regularized covariance')
    if isinstance(config.random_state, np.random.RandomState):
        msg = (f'Using custom random number generator in cross-validation '
               f'scheme: {config.random_state}')
    elif config.random_state is not None:
        msg = (f'Cross-validation random number generator will be seeded with '
               f'the user-defined value: {config.random_state}')
    else:
        msg = 'Using random number generator with default settings'

    print(msg)
    del msg

    cv = KFold(3, random_state=config.random_state, shuffle=True)
    cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', cv=cv,
                                 rank='info')
    cov.save(fname_cov)


def main():
    """Run cov."""
    parallel, run_func, _ = parallel_func(run_covariance, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    main()
