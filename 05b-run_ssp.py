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
from mne.preprocessing import compute_proj_ecg, compute_proj_eog


def run_ssp(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    print("  Loading one run to compute SSPs")

    # compute SSP on first run of raw
    run = config.runs[0]
    extension = run + '_sss_raw'
    raw_fname_in = op.join(meg_subject_dir,
                           config.base_fname.format(**locals()))

    extension = run + '_ssp-proj'
    proj_fname_out = op.join(meg_subject_dir,
                             config.base_fname.format(**locals()))

    print("Input: ", raw_fname_in)
    print("Output: ", proj_fname_out)

    raw = mne.io.read_raw_fif(raw_fname_in)
    # XXX : n_xxx should be options in config
    print("  Computing SSPs for ECG")
    ecg_projs, ecg_events = \
        compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=True)
    print("  Computing SSPs for EOG")
    eog_projs, eog_events = \
        compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, average=True)

    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ssp, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
