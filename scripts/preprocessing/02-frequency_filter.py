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

import sys
import itertools
import logging
import numpy as np
from typing import Optional, Union
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import mne
from mne.utils import BunchConst
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
    cfg,
    subject: str,
    run: Optional[str] = None,
    session: Optional[str] = None,
) -> None:
    """Filter data from a single subject."""
    if cfg.l_freq is None and cfg.h_freq is None:
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
    if cfg.use_maxwell_filter:
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
        h_freq=cfg.h_freq, l_freq=cfg.l_freq,
        h_trans_bandwidth=cfg.h_trans_bandwidth,
        l_trans_bandwidth=cfg.l_trans_bandwidth
    )
    resample(raw=raw, subject=subject, session=session, run=run,
             sfreq=cfg.resample_sfreq)

    raw.save(raw_fname_out, overwrite=True, split_naming='bids')
    if cfg.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
        raw.plot_psd(fmax=fmax)

    if cfg.process_er and run == cfg.runs[0]:
        # Ensure empty-room data has the same bad channel selection set as the
        # experimental data
        bads = raw.info['bads'].copy()
        del raw  # free memory

        bids_path_er = bids_path.copy().update(run=None, task='noise')
        if cfg.use_maxwell_filter:
            raw_er_fname_in = bids_path_er.copy().update(processing='sss')
            if raw_er_fname_in.copy().update(split='01').fpath.exists():
                raw_er_fname_in.update(split='01')
            msg = f'Reading empty-room recording: {raw_er_fname_in}'
            logger.info(gen_log_message(message=msg, step=2, subject=subject,
                                        session=session, run=run))
            raw_er = mne.io.read_raw_fif(raw_er_fname_in)
            raw_er.info['bads'] = bads
        else:
            raw_er = import_er_data(cfg=cfg, subject=subject, session=session,
                                    bads=bads, save=False)

        raw_er_fname_out = bids_path_er.copy().update(processing='filt')

        raw_er.load_data()
        filter(
            raw=raw_er, subject='emptyroom', session=session, run=run,
            h_freq=cfg.h_freq, l_freq=cfg.l_freq,
            h_trans_bandwidth=cfg.h_trans_bandwidth,
            l_trans_bandwidth=cfg.l_trans_bandwidth
        )
        resample(raw=raw_er, subject='emptyroom', session=session, run=run,
                 sfreq=cfg.resample_sfreq)

        raw_er.save(raw_er_fname_out, overwrite=True, split_naming='bids')
        if cfg.interactive:
            # Plot raw data and power spectral density.
            raw_er.plot(n_channels=50, butterfly=True)
            fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
            raw_er.plot_psd(fmax=fmax)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        process_er=config.process_er,
        runs=config.get_runs(subject=subject),
        use_maxwell_filter=config.use_maxwell_filter,
        proc=config.proc,
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.get_bids_root(),
        deriv_root=config.get_deriv_root(),
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        l_trans_bandwidth=config.l_trans_bandwidth,
        h_trans_bandwidth=config.h_trans_bandwidth,
        resample_sfreq=config.resample_sfreq,
        crop_runs=config.crop_runs,
        interactive=config.interactive,
        rename_events=config.rename_events,
        eeg_bipolar_channels=config.eeg_bipolar_channels,
        eeg_template_montage=config.eeg_template_montage,
        fix_stim_artifact=config.fix_stim_artifact,
        find_flat_channels_meg=config.find_flat_channels_meg,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        reference_run=config.get_mf_reference_run(),
        drop_channels=config.drop_channels,
        find_breaks=config.find_breaks,
        min_break_duration=config.min_break_duration,
        t_break_annot_start_after_previous_event=config.t_break_annot_start_after_previous_event,  # noqa:E501
        t_break_annot_stop_before_next_event=config.t_break_annot_stop_before_next_event,  # noqa:E501
    )
    return cfg


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
            cfg=get_config(subject),
            subject=subject,
            run=run,
            session=session
        ) for subject, run, session in sub_run_ses
    )

    msg = 'Completed 2: Frequency filtering'
    logger.info(gen_log_message(step=2, message=msg))


if __name__ == '__main__':
    main()
