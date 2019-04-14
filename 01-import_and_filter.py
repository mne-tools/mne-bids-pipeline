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
from warnings import warn

import config


def run_filter(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    n_raws = 0
    for run in config.runs:

        # read bad channels for run from config
        if run:
            bads = config.bads[subject][run]
        else:
            bads = config.bads[subject]

        extension = run + '_raw'
        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))

        extension = run + '_filt_raw'
        raw_fname_out = op.join(meg_subject_dir,
                                config.base_fname.format(**locals()))

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        if not op.exists(raw_fname_in):
            warn('Run %s not found for subject %s ' %
                 (raw_fname_in, subject))
            continue

        raw = mne.io.read_raw_fif(raw_fname_in,
                                  allow_maxshield=config.allow_maxshield,
                                  preload=True, verbose='error')

        # add bad channels
        raw.info['bads'] = bads
        print("added bads: ", raw.info['bads'])

        if config.set_channel_types is not None:
            raw.set_channel_types(config.set_channel_types)
        if config.rename_channels is not None:
            raw.rename_channels(config.rename_channels)

        # Band-pass the data channels (MEG and EEG)
        print("Filtering data between %s and %s (Hz)" %
              (config.l_freq, config.h_freq))
        raw.filter(
            config.l_freq, config.h_freq,
            l_trans_bandwidth=config.l_trans_bandwidth,
            h_trans_bandwidth=config.h_trans_bandwidth,
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')

        if config.resample_sfreq:
            print("Resampling data to %.1f Hz" % config.resample_sfreq)
            raw.resample(config.resample_sfreq, npad='auto')

        raw.save(raw_fname_out, overwrite=True)
        n_raws += 1

        if config.plot:
            # plot raw data
            raw.plot(n_channels=50, butterfly=True, group_by='position')

            # plot power spectral densitiy
            raw.plot_psd(area_mode='range', tmin=10.0, tmax=100.0,
                         fmin=0., fmax=50., average=True)

    if n_raws == 0:
        raise ValueError('No input raw data found.')


parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
