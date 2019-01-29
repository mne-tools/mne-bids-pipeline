"""
============================================
03. Extract events from the stimulus channel
============================================

The events are extracted from stimulus channel 'STI101'. The events are saved
to the subject's MEG directory.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_events(subject):
    print("processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]
    eve_fnames_out = [op.join(meg_subject_dir, '%s_audvis_filt-eve.fif' % subject)]

    for raw_fname_in, eve_fname_out in zip(raw_fnames_in, eve_fnames_out):
        raw = mne.io.read_raw_fif(raw_fname_in)
        events = mne.find_events(raw)

        print("subject: %s - file: %s" % (subject, raw_fname_in))

        mne.write_events(eve_fname_out, events)


parallel, run_func, _ = parallel_func(run_events, n_jobs=config.N_JOBS)
subjects_iterable = [config.subjects] if isinstance(config.subjects, str) else config.subjects 
parallel(run_func(subject) for subject in subjects_iterable)