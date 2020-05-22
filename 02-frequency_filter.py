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
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def run_filter(subject, run=None, session=None):
    """Filter data from a single subject."""
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

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
    if config.use_maxwell_filter:
        raw_fname_in = op.join(deriv_path, bids_basename + '_sss_raw.fif')
    else:
        raw_fname_in = op.join(deriv_path, bids_basename + '_nosss_raw.fif')

    raw_fname_out = op.join(deriv_path, bids_basename + '_filt_raw.fif')

    msg = f'Input: {raw_fname_in}, Output: {raw_fname_out}'
    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run,))

    raw = mne.io.read_raw_fif(raw_fname_in)
    raw.load_data()

    # Band-pass the data channels (MEG and EEG)
    msg = f'Filtering data between {config.l_freq} and {config.h_freq} (Hz)'
    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run,))

    raw.filter(config.l_freq, config.h_freq,
               l_trans_bandwidth=config.l_trans_bandwidth,
               h_trans_bandwidth=config.h_trans_bandwidth,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin'
               )

    if config.resample_sfreq:
        msg = f'Resampling data to {config.resample_sfreq:.1f} Hz'
        logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                    session=session, run=run,))
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
    msg = 'Running Step 2: Frequency filtering'
    logger.info(gen_log_message(step=2, message=msg))

    parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
    parallel(run_func(subject, run, session) for subject, run, session in
             itertools.product(config.get_subjects(), config.get_runs(),
                               config.get_sessions()))

    msg = 'Completed 2: Frequency filtering'
    logger.info(gen_log_message(step=2, message=msg))


if __name__ == '__main__':
    main()
