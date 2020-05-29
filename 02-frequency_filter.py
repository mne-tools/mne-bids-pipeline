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

If config.interactive = True plots raw data and power spectral density.

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


def run_filter(subject, run=None, session=None):
    """Filter data from a single subject."""
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    shared_basename_kws = dict(subject=subject,
                               session=session,
                               task=config.get_task(),
                               acquisition=config.acq,
                               processing=config.proc,
                               recording=config.rec,
                               space=config.space)

    bids_basename = make_bids_basename(run=run, **shared_basename_kws)
    bids_er_out_basename = make_bids_basename(**shared_basename_kws)

    # Prepare a name to save the data
    if config.use_maxwell_filter:
        raw_fname_in = op.join(deriv_path, f'{bids_basename}_sss_raw.fif')
        raw_er_fname_in = op.join(deriv_path,
                                  f'{bids_basename}_emptyroom_sss_raw.fif')
    else:
        raw_fname_in = op.join(deriv_path, f'{bids_basename}_nosss_raw.fif')
        raw_er_fname_in = op.join(
            deriv_path,
            f'{bids_basename}_emptyroom_nosss_raw.fif')

    raw_fname_out = op.join(deriv_path, f'{bids_basename}_filt_raw.fif')
    raw_er_fname_out = op.join(
        deriv_path,
        f'{bids_er_out_basename}_emptyroom_filt_raw.fif')

    msg = f'Input: {raw_fname_in}, Output: {raw_fname_out}'
    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run,))

    raw = mne.io.read_raw_fif(raw_fname_in)
    raw.load_data()

    # Band-pass the data channels (MEG and EEG)
    msg = (f'Filtering experimental data between {config.l_freq} and '
           f'{config.h_freq} Hz')
    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run))

    filter_kws = dict(l_freq=config.l_freq, h_freq=config.h_freq,
                      l_trans_bandwidth=config.l_trans_bandwidth,
                      h_trans_bandwidth=config.h_trans_bandwidth,
                      filter_length='auto', phase='zero', fir_window='hamming',
                      fir_design='firwin')
    raw.filter(**filter_kws)

    if config.noise_cov == 'emptyroom':
        msg = f'Filtering empty-room recording.'
        logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                    session=session, run=run,))
        raw_er = mne.io.read_raw_fif(raw_er_fname_in)
        raw_er.load_data()
        raw_er.filter(**filter_kws)

    if config.resample_sfreq:
        msg = f'Resampling experimental data to {config.resample_sfreq:.1f} Hz'
        logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                    session=session, run=run,))
        raw.resample(config.resample_sfreq, npad='auto')

        if config.noise_cov == 'emptyroom':
            msg = f'Resampling empty-room recording.'
            logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                        session=session, run=run,))
            raw_er.resample(config.resample_sfreq, npad='auto')

    raw.save(raw_fname_out, overwrite=True)
    if config.noise_cov == 'emptyroom':
        raw_er.save(raw_er_fname_out, overwrite=True)

    if config.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        raw.plot_psd()

        if config.noise_cov == 'emptyroom':
            raw_er.plot(n_channels=50, butterfly=True)
            raw_er.plot_psd()


@failsafe_run(on_error=on_error)
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
