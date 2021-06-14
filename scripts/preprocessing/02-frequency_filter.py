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

from typing import Optional, Union, Literal
import itertools
import logging
import numpy as np

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import (gen_log_message, on_error, failsafe_run,
                    import_experimental_data, import_er_data)

logger = logging.getLogger('mne-bids-pipeline')


def filter(
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    l_freq: Optional[float],
    h_freq: Optional[float],
    l_trans_bandwidth: Optional[Union[float, Literal['auto']]],
    h_trans_bandwidth: Optional[Union[float, Literal['auto']]]
) -> None:
    """Filter data channels (MEG and EEG)."""
    if l_freq is None and h_freq is None:
        return

    data_type = 'empty-room' if subject == 'emptyroom' else 'experimental'

    if l_freq is not None and h_freq is None:
        msg = (f'High-pass filtering {data_type} data; lower bound: '
               f'{l_freq} Hz')
    elif l_freq is None and h_freq is not None:
        msg = (f'Low-pass filtering {data_type} data; upper bound: '
               f'{h_freq} Hz')
    elif l_freq is not None and h_freq is not None:
        msg = (f'Band-pass filtering {data_type} data; range: '
               f'{l_freq} – {h_freq} Hz')

    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run))

    raw.filter(l_freq=l_freq, h_freq=h_freq,
               l_trans_bandwidth=l_trans_bandwidth,
               h_trans_bandwidth=h_trans_bandwidth,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin')


def resample(
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    sfreq: Optional[float]
) -> None:
    if not sfreq:
        return

    data_type = 'empty-room' if subject == 'emptyroom' else 'experimental'
    msg = f'Resampling {data_type} data to {sfreq:.1f} Hz'
    logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                session=session, run=run,))
    raw.resample(sfreq, npad='auto')


def filter_data(
    *,
    cfg: dict,
    subject: str,
    run: Optional[str] = None,
    session: Optional[str] = None,
    l_freq: Optional[float],
    h_freq: Optional[float],
    l_trans_bandwidth: Optional[Union[float, Literal['auto']]],
    h_trans_bandwidth: Optional[Union[float, Literal['auto']]],
    resample_sfreq: Optional[float]
) -> None:
    """Filter data from a single subject."""
    if l_freq is None and h_freq is None:
        return

    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    bids_path = BIDSPath(subject=subject,
                         run=run,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         processing=cfg.proc,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix='raw',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    # Create paths for reading and writing the filtered data.
    if config.use_maxwell_filter:
        raw_fname_in = bids_path.copy().update(processing='sss')
        if raw_fname_in.copy().update(split='01').fpath.exists():
            raw_fname_in.update(split='01')
        msg = f'Reading: {raw_fname_in}'
        logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                    session=session, run=run))
        raw = mne.io.read_raw_fif(raw_fname_in)
    else:
        raw = import_experimental_data(cfg=cfg,
                                       subject=subject, session=session,
                                       run=run, save=False)

    raw_fname_out = bids_path.copy().update(processing='filt')

    raw.load_data()
    filter(
        raw=raw, subject=subject, session=session, run=run,
        h_freq=h_freq, l_freq=l_freq,
        h_trans_bandwidth=h_trans_bandwidth,
        l_trans_bandwidth=l_trans_bandwidth
    )
    resample(raw=raw, subject=subject, session=session, run=run,
             sfreq=resample_sfreq)

    raw.save(raw_fname_out, overwrite=True, split_naming='bids')
    if config.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        fmax = 1.5 * config.h_freq if config.h_freq is not None else np.inf
        raw.plot_psd(fmax=fmax)

    if config.process_er and run == config.get_runs(subject)[0]:
        # Ensure empty-room data has the same bad channel selection set as the
        # experimental data
        bads = raw.info['bads'].copy()
        del raw  # free memory

        bids_path_er = bids_path.copy().update(run=None, task='noise')
        if config.use_maxwell_filter:
            raw_er_fname_in = bids_path_er.copy().update(processing='sss')
            if raw_er_fname_in.copy().update(split='01').fpath.exists():
                raw_er_fname_in.update(split='01')
            msg = f'Reading empty-room recording: {raw_er_fname_in}'
            logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                        session=session, run=run))
            raw_er = mne.io.read_raw_fif(raw_er_fname_in)
            raw_er.info['bads'] = bads
        else:
            raw_er = import_er_data(subject=subject, session=session,
                                    bads=bads, save=False)

        raw_er_fname_out = bids_path_er.copy().update(processing='filt')

        raw_er.load_data()
        filter(
            raw=raw_er, subject=subject, session=session, run=run,
            h_freq=h_freq, l_freq=l_freq,
            h_trans_bandwidth=h_trans_bandwidth,
            l_trans_bandwidth=l_trans_bandwidth
        )
        resample(raw=raw_er, subject=subject, session=session, run=run,
                 sfreq=resample_sfreq)

        raw_er.save(raw_er_fname_out, overwrite=True, split_naming='bids')
        if config.interactive:
            # Plot raw data and power spectral density.
            raw_er.plot(n_channels=50, butterfly=True)
            fmax = 1.5 * config.h_freq if config.h_freq is not None else np.inf
            raw_er.plot_psd(fmax=fmax)


@failsafe_run(on_error=on_error)
def main():
    """Run filter."""
    msg = 'Running Step 2: Frequency filtering'
    logger.info(gen_log_message(step=2, message=msg))

    parallel, run_func, _ = parallel_func(filter_data, n_jobs=config.N_JOBS)

    # Enabling different runs for different subjects
    sub_run_ses = []
    for subject in config.get_subjects():
        sub_run_ses += list(itertools.product(
            [subject],
            config.get_runs(subject=subject),
            config.get_sessions()))

    parallel(
        run_func(
            subject=subject,
            run=run,
            session=session,
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            l_trans_bandwidth=config.l_trans_bandwidth,
            h_trans_bandwidth=config.h_trans_bandwidth,
            resample_sfreq=config.resample_sfreq
        ) for subject, run, session in sub_run_ses
    )

    msg = 'Completed 2: Frequency filtering'
    logger.info(gen_log_message(step=2, message=msg))


if __name__ == '__main__':
    main()
