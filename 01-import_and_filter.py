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
for more. The filtered data are saved to separate files to the subject's 'MEG'
directory.

If config.plot = True plots raw data and power spectral density.
"""  # noqa: E501

import os
import os.path as op
import glob

from mne.parallel import parallel_func
from mne_bids.read import reader as mne_bids_readers
from mne_bids import make_bids_basename, read_raw_bids

import config


def run_filter(subject):
    """Filter data from a single subject."""
    print('\nProcessing subject: {}\n{}'
          .format(subject, '-' * (20 + len(subject))))

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `ses` is optional
    if config.ses:
        subject_path = op.join(subject_path, 'ses-{}'.format(config.ses))
    # `kind` is mandatory, e.g., "eeg", "meg", "anat"
    subject_path = op.join(subject_path, config.kind)
    data_dir = op.join(config.bids_root, subject_path)

    bids_basename = make_bids_basename(subject=subject,
                                       session=config.ses,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=config.run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    # Find the data file
    search_str = op.join(data_dir, bids_basename) + '_' + config.kind + '*'
    fnames = sorted(glob.glob(search_str))
    fnames = [f for f in fnames
              if op.splitext(f)[1] in mne_bids_readers]

    if len(fnames) == 1:
        bids_fpath = fnames[0]
    elif len(fnames) == 0:
        raise ValueError('Could not find input data file matching: '
                         '"{}"'.format(search_str))
    elif len(fnames) > 1:
        raise ValueError('Expected to find a single input data file: "{}" '
                         ' but found:\n\n{}'
                         .format(search_str, fnames))

    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv
    _, bids_fname = op.split(bids_fpath)
    raw = read_raw_bids(bids_fname, config.bids_root)
    raw.load_data()

    # Band-pass the data channels (MEG and EEG)
    print('Filtering data between {} and {} (Hz)'
          .format(config.l_freq, config.h_freq))

    raw.filter(config.l_freq, config.h_freq,
               l_trans_bandwidth=config.l_trans_bandwidth,
               h_trans_bandwidth=config.h_trans_bandwidth,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin'
               )

    if config.resample_sfreq:
        print('Resampling data to {:.1f} Hz'.format(config.resample_sfreq))

        raw.resample(config.resample_sfreq, npad='auto')

    # Prepare a name to save the data
    fpath_out = op.join(config.bids_root, 'derivatives', subject_path)
    if not op.exists(fpath_out):
        os.makedirs(fpath_out)
    fname_out = op.join(fpath_out, bids_basename + '_filt_raw.fif')

    raw.save(fname_out, overwrite=True)

    if config.plot:
        # plot raw data
        raw.plot(n_channels=50, butterfly=True)

        # plot power spectral densitiy
        raw.plot_psd(area_mode='range', tmin=10.0, tmax=100.0,
                     fmin=0., fmax=50., average=True)


parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
