"""
============================================
03. Extract events from the stimulus channel
============================================

Here, all events present in the stimulus channel indicated in
config.stim_channel are extracted.
The events are saved to the subject's MEG directory.
This is done early in the pipeline to avoid distorting event-time,
for instance by resampling.
"""

import os.path as op
import itertools

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


def run_events(subject, run=None, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    # Prepare a name to save the data
    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          'mne-study-template', subject_path)
    if config.use_maxwell_filter:
        raw_fname_in = \
            op.join(fpath_deriv, bids_basename + '_sss_raw.fif')
    else:
        raw_fname_in = \
            op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

    eve_fname_out = op.join(fpath_deriv, bids_basename + '-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname_in)
    events, event_id = mne.events_from_annotations(raw)

    if config.trigger_time_shift:
        events = mne.event.shift_time_events(events,
                                             np.unique(events[:, 2]),
                                             config.trigger_time_shift,
                                             raw.info['sfreq'])

    print("Input: ", raw_fname_in)
    print("Output: ", eve_fname_out)

    mne.write_events(eve_fname_out, events)

    if config.plot:
        # plot events
        mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp)
        plt.show()


parallel, run_func, _ = parallel_func(run_events, n_jobs=config.N_JOBS)
parallel(run_func(subject, run, session) for subject, run, session in
         itertools.product(config.subjects_list, config.runs, config.sessions))
