"""
===========
05. Run SSP
===========

SSP
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_ssp(subject):
    print("processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]
    proj_fnames_out = [op.join(meg_subject_dir, '%s_audvis_filt-proj.fif' % subject)]

    print("  Loading runs")
    for raw_fname_in, proj_fname_out in zip(raw_fnames_in, proj_fnames_out):
        # XXX TODO
        raw = mne.io.read_raw_fif(raw_fnames_in)

    # XXX

# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ssp, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects)
