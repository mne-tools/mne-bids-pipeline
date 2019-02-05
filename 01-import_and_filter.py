"""
===========================
01. Filter using MNE-python
===========================

The data are bandpass filtered to the frequencies defined in config.py
(config.h_freq - config.l_freq Hz) using linear-phase fir filter with
delay compensation.
The transition bandwidth is automatically defined. See
`Background information on filtering
<http://mne-tools.github.io/dev/auto_tutorials/plot_background_filtering.html>`_
for more. The filtered data are saved to separate files to the subject's'MEG'
directory.
If config.plot = True plots raw data and power spectral density.
"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_filter(subject):
    print("processing subject: %s" % subject)
    # XXX : put the study-specific names in the config file
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_raw.fif' % subject)]
    raw_fnames_out = [
        op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]

    for raw_fname_in, raw_fname_out in zip(raw_fnames_in, raw_fnames_out):
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True, verbose='error')

        # add bad channels from config
        # XXX allow to add bad channels per run
        raw.info['bads'] = config.bads[subject]
        print("added bads: ", raw.info['bads'])

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

        raw.save(raw_fname_out, overwrite=True)

        # XXX if we add multiple runs, this should probably plot an appended
        # version of the data
        if config.plot:
            # plot raw data
            figure = raw.plot(n_channels=50, butterfly=True,
                              group_by='position')
            figure.show()

            # plot power spectral densitiy
            figure = raw.plot_psd(area_mode='range', tmin=10.0, tmax=100.0,
                                  fmin=0., fmax=50., average=True)
            figure.show()


parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
