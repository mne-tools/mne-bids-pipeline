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
    print("processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)
    proj_fnames_out = op.join(meg_subject_dir, '%s_audvis_filt-proj.fif' % subject)

    print("  Loading runs")
    for raw_fname_in, proj_fname_out in zip(raw_fnames_in, proj_fnames_out):
        # XXX TODO
        raw = mne.io.read_raw_fif(raw_fnames_in)
        projs, events = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=True)
        ecg_projs = projs[-2:]
        projs, events = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, average=True)
        eog_projs = projs[-3:]

        raw.info['projs'] += eog_projs + ecg_projs
        raw.apply_proj()
        raw.save(proj_fnames_out, overwrite=True)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ssp, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
