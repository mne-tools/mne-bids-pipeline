"""
============================================
03. Extract events from the stimulus channel
============================================

Here, all events present in the stimulus channel 'STI101' are extracted. 
The events are saved to the subject's MEG directory.
This is done early in the pipeline to avoid distorting event-time, for instance
by resampling.  
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_events(subject):
    print("processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [
        op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]
    eve_fnames_out = [
        op.join(meg_subject_dir, '%s_audvis_filt-eve.fif' % subject)]

    for raw_fname_in, eve_fname_out in zip(raw_fnames_in, eve_fnames_out):
        raw = mne.io.read_raw_fif(raw_fname_in)
        events = mne.find_events(raw, stim_channel=config.stim_channel)

        print("subject: %s - file: %s" % (subject, raw_fname_in))

        mne.write_events(eve_fname_out, events)

        if config.plot:
            # plot events
            figure = mne.viz.plot_events(events)
            figure.show()


parallel, run_func, _ = parallel_func(run_events, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
