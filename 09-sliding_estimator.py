"""
=================
Sliding estimator
=================

A sliding estimator fits a logistic legression model for every time point.
In this example, we contrast the condition 'famous' against 'scrambled'
using this approach. The end result is an averaging effect across sensors.
The contrast across different sensors are combined into a single plot.

"""  # noqa: E501

###############################################################################
# Let us first import the libraries

import os.path as op

import numpy as np
from scipy.io import savemat

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import config

###############################################################################
# Then we write a function to do time decoding on one subject


def run_time_decoding(subject, condition1, condition2):
    print("processing subject: %s (%s vs %s)"
          % (subject, condition1, condition2))

    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    epochs = mne.read_epochs(fname_in)

    # We define the epochs and the labels
    epochs = mne.concatenate_epochs([epochs[condition1],
                                     epochs[condition2]])
    epochs.apply_baseline()

    # Get the data and labels
    X = epochs.get_data()
    n_cond1 = len(epochs[condition1])
    n_cond2 = len(epochs[condition2])
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]

    # Use AUC because chance level is same regardless of the class balance
    se = SlidingEstimator(
        make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear',
                                         random_state=config.random_state)),
        scoring=config.decoding_metric, n_jobs=config.N_JOBS)
    cv = StratifiedKFold(random_state=config.random_state,
                         n_splits=config.decoding_n_splits)
    scores = cross_val_multiscore(se, X=X, y=y, cv=cv)

    # let's save the scores now
    a_vs_b = '%s_vs_%s' % (condition1, condition2)
    a_vs_b = a_vs_b.replace(op.sep, '')
    fname_td = op.join(meg_subject_dir, '%s_%s_%s_%s.mat'
                       % (subject, config.study_name, a_vs_b,
                          config.decoding_metric))
    savemat(fname_td, {'scores': scores, 'times': epochs.times})

# Here we go parallel inside the :class:`mne.decoding.SlidingEstimator`
# so we don't dispatch manually to multiple jobs.

for subject in config.subjects_list:
    for conditions in config.decoding_conditions:
        run_time_decoding(subject, *conditions)
