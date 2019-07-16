"""
==================================
08. Baseline covariance estimation
==================================

Covariance matrices are computed and saved.
"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

from sklearn.model_selection import KFold

import config


def run_covariance(subject):
    print("Processing subject: %s%s")
    subject_path = op.join('sub-{}'.format(subject), config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=config.ses,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=config.run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    if config.use_ica or config.use_ssp:
        extension = '_cleaned-epo'
    else:
        extension = '-epo'

    fpath_deriv = op.join(config.bids_root, 'derivatives', subject_path)
    fname_epo = \
        op.join(fpath_deriv, bids_basename + '%s.fif' % extension)

    fname_cov = \
        op.join(fpath_deriv, bids_basename + '-cov.fif')

    print("Input: ", fname_epo)
    print("Output: ", fname_cov)

    epochs = mne.read_epochs(fname_epo, preload=True)

    print('  Computing regularized covariance')
    cv = KFold(3, random_state=config.random_state)  # make cv deterministic
    cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', cv=cv)
    cov.save(fname_cov)


parallel, run_func, _ = parallel_func(run_covariance, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
