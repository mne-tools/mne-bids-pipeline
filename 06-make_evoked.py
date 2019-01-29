"""
===============
06. Evoked data
===============

The evoked data sets are created by averaging different conditions.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_evoked(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    fname_epo = op.join(meg_subject_dir, '%s-epo.fif' % subject)
    fname_ave = op.join(meg_subject_dir, '%s-ave.fif' % subject)

    print('  Creating evoked datasets')
    epochs = mne.read_epochs(fname_epo, preload=True)

    evokeds = []
    for condition in config.conditions:
        evokeds.append(epochs[condition].average())

    mne.evoked.write_evokeds(fname_ave, evokeds)


parallel, run_func, _ = parallel_func(run_evoked, n_jobs=config.N_JOBS)

subjects_iterable = [config.subjects] if isinstance(config.subjects, str) else config.subjects 
parallel(run_func(subject) for subject in subjects_iterable)