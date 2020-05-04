"""
==========================
02. Apply frequency filter
==========================

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

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


def run_filter(subject, run=None, session=None):
    """Filter data from a single subject."""
    print('\nProcessing subject: {}\n{}'
          .format(subject, '-' * (20 + len(subject))))

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    # Prepare a name to save the data
    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
    if config.use_maxwell_filter:
        raw_fname_in = op.join(fpath_deriv, bids_basename + '_sss_raw.fif')
    else:
        raw_fname_in = op.join(fpath_deriv, bids_basename + '_nosss_raw.fif')

    raw_fname_out = op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

    print("Input: ", raw_fname_in)
    print("Output: ", raw_fname_out)

    raw = mne.io.read_raw_fif(raw_fname_in)
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

    raw.save(raw_fname_out, overwrite=True)

    if config.plot:
        # plot raw data
        raw.plot(n_channels=50, butterfly=True)

        # plot power spectral densitiy
        raw.plot_psd(area_mode='range', tmin=10.0, tmax=100.0,
                     fmin=0., fmax=50., average=True)


def main():
    """Run filter."""
    parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
    parallel(run_func(subject, run, session) for subject, run, session in
             itertools.product(config.get_subjects(), config.get_runs(),
                               config.get_sessions()))


if __name__ == '__main__':
    main()
