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

import itertools
import logging
import numpy as np

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def run_filter(subject, run=None, session=None):
    """Filter data from a single subject."""

    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    bids_path = BIDSPath(subject=subject,
                         run=run,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         processing=config.proc,
                         recording=config.rec,
                         space=config.space,
                         suffix='raw',
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    # Prepare a name to save the data
    raw_fname_in = bids_path.copy()
    raw_er_fname_in = bids_path.copy().update(task='noise', run=None)

    if config.use_maxwell_filter:
        raw_fname_in = raw_fname_in.update(processing='sss')
        raw_er_fname_in = raw_er_fname_in.update(processing='sss')

    if raw_fname_in.copy().update(split='01').fpath.exists():
        raw_fname_in.update(split='01')
    if raw_er_fname_in.copy().update(split='01').fpath.exists():
        raw_er_fname_in.update(split='01')

    raw_fname_out = bids_path.copy().update(processing='filt')
    raw_er_fname_out = bids_path.copy().update(run=None, processing='filt',
                                               task='noise')

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

    if config.process_er:
        msg = 'Filtering empty-room recording.'
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

        if config.process_er:
            msg = 'Resampling empty-room recording.'
            logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                        session=session, run=run,))
            raw_er.resample(config.resample_sfreq, npad='auto')

    raw.save(raw_fname_out, overwrite=True, split_naming='bids')
    if config.process_er:
        raw_er.save(raw_er_fname_out, overwrite=True, split_naming='bids')

    if config.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        fmax = 1.5 * config.h_freq if config.h_freq is not None else np.inf
        raw.plot_psd(fmax=fmax)

        if config.process_er:
            raw_er.plot(n_channels=50, butterfly=True)
            raw_er.plot_psd(fmax=fmax)


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
