"""
==================================
08. Baseline covariance estimation
==================================

Covariance matrices are computed and saved.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

from sklearn.model_selection import KFold

import config


def run_covariance(subject):
    print("Processing subject: %s%s")
    meg_subject_dir = op.join(config.meg_dir, subject)
    fname_epochs = op.join(meg_subject_dir,
                            config.base_epochs_fname.format(**locals()))
    fname_cov = op.join(meg_subject_dir, '%s-cov.fif' % subject)
    print('  Computing regularized covariance')
    epochs = mne.read_epochs(fname_epochs, preload=True)
    cv = KFold(3, random_state=config.random_state)  # make cv deterministic
    cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', cv=cv)
    cov.save(fname_cov)


parallel, run_func, _ = parallel_func(run_covariance, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
