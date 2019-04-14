"""
===============
06b. Apply SSP
===============

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def apply_ssp(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    # load epochs to reject ICA components
    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))
    epochs = mne.read_epochs(fname_in, preload=True)

    extension = '_cleaned-epo'
    fname_out = op.join(meg_subject_dir,
                        config.base_fname.format(**locals()))

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    run = config.runs[0]
    extension = run + '_ssp-proj'
    proj_fname_in = op.join(meg_subject_dir,
                            config.base_fname.format(**locals()))

    print("Reading SSP projections from : %s" % proj_fname_in)

    projs = mne.read_proj(proj_fname_in)
    epochs.add_proj(projs).apply_proj()

    print('Saving epochs')
    epochs.save(fname_out)


parallel, run_func, _ = parallel_func(apply_ssp, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
