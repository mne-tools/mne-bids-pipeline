"""
===========================
01. Filter using MNE-python
===========================

The data are bandpass filtered (1 - 40 Hz) using linear-phase fir filter with
delay compensation. For the lowpass filter the transition bandwidth is
automatically defined. See
`Background information on filtering <http://mne-tools.github.io/dev/auto_tutorials/plot_background_filtering.html>`_
for more. The filtered data are saved to separate files to the subject's'MEG'
directory.
"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func

import config

# set to True if you want to plot the raw data
do_plot = False

def run_filter(subject):
    print("processing subject: %s" % subject)
    # XXX : put the study-specific names in the config file
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_raw.fif' % subject)]
    raw_fnames_out = [op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]

    for raw_fname_in, raw_fname_out in zip(raw_fnames_in, raw_fnames_out):
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True, verbose='error')
        # XXX : to add to config.py
        if config.set_channel_types is not None:
            raw.set_channel_types(config.set_channel_types)
        if config.rename_channels is not None:
            raw.rename_channels(config.rename_channels)

        # Band-pass the data channels (MEG and EEG)
        raw.filter(
            config.l_freq, config.h_freq,
            l_trans_bandwidth=config.l_trans_bandwidth,
            h_trans_bandwidth=config.h_trans_bandwidth,
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')
        
        if do_plot:
            figure = raw.plot(n_channels = 50,butterfly=True, group_by='position') 
    
        raw.save(raw_fname_out, overwrite=True)


parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)

subjects_iterable = [config.subjects] if isinstance(config.subjects, str) else config.subjects 
parallel(run_func(subject) for subject in subjects_iterable)
