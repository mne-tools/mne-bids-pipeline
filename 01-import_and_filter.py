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
    print("processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_fnames_in = list()
    raw_fnames_out = list()

    # input files
    if config.runs:
        base_raw_fname_in = '{subject}_{study_name}_{run}_raw.fif'
        base_raw_fname_out = '{subject}_{study_name}_{run}_filt_raw.fif'

        for run in config.runs:
            raw_fnames_in.append(base_raw_fname_in.format(run=run,
                                                          study_name=config.study_name,
                                                          subject=subject))

            raw_fnames_out.append(base_raw_fname_out.format(run=run,
                                                            study_name=config.study_name,
                                                            subject=subject))

    else:
        base_raw_fname_in = '{subject}_{study_name}_raw.fif'
        base_raw_fname_out = '{subject}_{study_name}_filt_raw.fif'

        raw_fnames_in.append(base_raw_fname_in.format(
            study_name=config.study_name, subject=subject))
        raw_fnames_out.append(base_raw_fname_out.format(
            study_name=config.study_name, subject=subject))

    raws = []
    print("Try loading %d files" % len(raw_fnames_in))
    for raw_fname_in, raw_fname_out in zip(raw_fnames_in, raw_fnames_out):
        print(raw_fname_in)

        raw_fname_path = op.join(meg_subject_dir, raw_fname_in)
        if not op.exists(raw_fname_path):
            if not raws:
                raise ValueError('Cannot find ' + raw_fname_path)
            else:
                warn('Run %s not found for subject %s ' %
                     (raw_fname_in, subject))
                continue

        raw = mne.io.read_raw_fif(raw_fname_path,
                                  preload=True, verbose='error')

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

        raw.save(op.join(meg_subject_dir, raw_fname_out), overwrite=True)
        raws.append(raw)

    if config.plot:

        # concatenate runs for plotting
        raw_all = mne.concatenate_raws(raws)

        # plot raw data
        figure = raw_all.plot(n_channels=50, butterfly=True,
                              group_by='position')
        figure.show()

        # plot power spectral densitiy
        figure = raw_all.plot_psd(area_mode='range', tmin=10.0, tmax=100.0,
                                  fmin=0., fmax=50., average=True)
        figure.show()


parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
