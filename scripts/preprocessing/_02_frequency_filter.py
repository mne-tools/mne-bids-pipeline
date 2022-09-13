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

To save space, the raw data can be resampled.

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
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

import config
from config import (gen_log_kwargs, failsafe_run,
                    import_experimental_data, import_er_data, import_rest_data,
                    _update_for_splits)
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


def get_input_fnames_frequency_filter(**kwargs):
    """Get paths of files required by filter_data function."""
    cfg = kwargs['cfg']
    subject = kwargs['subject']
    session = kwargs['session']
    run = kwargs['run']

    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    path_kwargs = dict(
        subject=subject,
        run=run,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        processing=cfg.proc,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        check=False
    )
    if cfg.use_maxwell_filter:
        path_kwargs['root'] = cfg.deriv_root
        path_kwargs['suffix'] = 'raw'
        path_kwargs['extension'] = '.fif'
    else:
        path_kwargs['root'] = cfg.bids_root
    bids_path_in = BIDSPath(**path_kwargs)

    if cfg.use_maxwell_filter:
        bids_path_in.update(processing="sss")

    in_files = dict()
    in_files[f'raw_run-{run}'] = bids_path_in
    _update_for_splits(in_files, f'raw_run-{run}', single=True)

    if (cfg.process_er or config.noise_cov == 'rest') and run == cfg.runs[0]:
        noise_task = "rest" if config.noise_cov == "rest" else "noise"
        if cfg.use_maxwell_filter:
            raw_noise_fname_in = bids_path_in.copy().update(
                run=None, task=noise_task
            )
            in_files["raw_noise"] = raw_noise_fname_in
            _update_for_splits(in_files, "raw_noise", single=True)
        else:
            if config.noise_cov == 'rest':
                in_files["raw_rest"] = bids_path_in.copy().update(
                    run=None, task=noise_task)
            else:
                assert config.noise_cov == 'noise'
                ref_bids_path = bids_path_in.copy().update(
                    run=cfg.mf_reference_run,
                    extension='.fif',
                    suffix='meg',
                    root=cfg.bids_root,
                    check=True
                )
                in_files["raw_er"] = ref_bids_path.find_empty_room()

    return in_files


def filter(
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    l_freq: Optional[float],
    h_freq: Optional[float],
    l_trans_bandwidth: Optional[Union[float, Literal['auto']]],
    h_trans_bandwidth: Optional[Union[float, Literal['auto']]],
    data_type: Literal['experimental', 'empty-room', 'resting-state']
) -> None:
    """Filter data channels (MEG and EEG)."""
    if l_freq is not None and h_freq is None:
        msg = (f'High-pass filtering {data_type} data; lower bound: '
               f'{l_freq} Hz')
    elif l_freq is None and h_freq is not None:
        msg = (f'Low-pass filtering {data_type} data; upper bound: '
               f'{h_freq} Hz')
    elif l_freq is not None and h_freq is not None:
        msg = (f'Band-pass filtering {data_type} data; range: '
               f'{l_freq} – {h_freq} Hz')
    else:
        msg = (f'Not applying frequency filtering to {data_type} data.')

    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run))

    if l_freq is None and h_freq is None:
        return

    raw.filter(l_freq=l_freq, h_freq=h_freq,
               l_trans_bandwidth=l_trans_bandwidth,
               h_trans_bandwidth=h_trans_bandwidth,
               filter_length='auto', phase='zero', fir_window='hamming',
               fir_design='firwin', n_jobs=1)


def resample(
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    sfreq: Optional[float],
    data_type: Literal['experimental', 'empty-room', 'resting-state']
) -> None:
    if not sfreq:
        return

    msg = f'Resampling {data_type} data to {sfreq:.1f} Hz'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run,))
    raw.resample(sfreq, npad='auto')


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_frequency_filter)
def filter_data(
    *,
    cfg,
    subject: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    in_files: Optional[dict] = None
) -> None:
    """Filter data from a single subject."""

    out_files = dict()
    bids_path = in_files[f"raw_run-{run}"]

    # Create paths for reading and writing the filtered data.
    if cfg.use_maxwell_filter:
        msg = f'Reading: {bids_path.basename}'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        raw = mne.io.read_raw_fif(bids_path)
    else:
        raw = import_experimental_data(bids_path_in=bids_path,
                                       cfg=cfg)

    out_files['raw_filt'] = bids_path.copy().update(
        root=cfg.deriv_root, processing='filt', extension='.fif',
        suffix='raw', split=None)
    raw.load_data()
    filter(
        raw=raw, subject=subject, session=session, run=run,
        h_freq=cfg.h_freq, l_freq=cfg.l_freq,
        h_trans_bandwidth=cfg.h_trans_bandwidth,
        l_trans_bandwidth=cfg.l_trans_bandwidth,
        data_type='experimental'
    )
    resample(raw=raw, subject=subject, session=session, run=run,
             sfreq=cfg.resample_sfreq, data_type='experimental')

    raw.save(out_files['raw_filt'], overwrite=True, split_naming='bids',
             split_size=cfg._raw_split_size)
    _update_for_splits(out_files, 'raw_filt')
    if cfg.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
        raw.plot_psd(fmax=fmax)

    del raw

    if (cfg.process_er or config.noise_cov == 'rest') and run == cfg.runs[0]:
        data_type = ('resting-state' if config.noise_cov == 'rest'
                     else 'empty-room')

        if cfg.use_maxwell_filter:
            bids_path_noise = in_files["raw_noise"]
            msg = (f'Reading {data_type} recording: '
                   f'{bids_path_noise.basename}')
            logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                         session=session))
            raw_noise = mne.io.read_raw_fif(in_files['raw_noise'])
        elif data_type == 'empty-room':
            bids_path_noise = in_files['raw_er']
            raw_noise = import_er_data(
                cfg=cfg,
                bids_path_er_in=bids_path_noise,
                bids_path_ref_in=None,
            )
        else:
            bids_path_noise = in_files['raw_rest']
            raw_noise = import_rest_data(
                cfg=cfg,
                bids_path_in=bids_path_noise
            )

        out_files['raw_noise_filt'] = \
            bids_path_noise.copy().update(
                root=cfg.deriv_root, processing='filt', extension='.fif',
                suffix='raw', split=None)

        raw_noise.load_data()
        filter(
            raw=raw_noise, subject=subject, session=session, run=None,
            h_freq=cfg.h_freq, l_freq=cfg.l_freq,
            h_trans_bandwidth=cfg.h_trans_bandwidth,
            l_trans_bandwidth=cfg.l_trans_bandwidth,
            data_type=data_type
        )
        resample(raw=raw_noise, subject=subject, session=session, run=None,
                 sfreq=cfg.resample_sfreq, data_type=data_type)

        raw_noise.save(
            out_files['raw_noise_filt'], overwrite=True, split_naming='bids',
            split_size=cfg._raw_split_size,
        )
        _update_for_splits(out_files, 'raw_noise_filt')
        if cfg.interactive:
            # Plot raw data and power spectral density.
            raw_noise.plot(n_channels=50, butterfly=True)
            fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
            raw_noise.plot_psd(fmax=fmax)

    return out_files


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        reader_extra_params=config.reader_extra_params,
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
        stim_artifact_tmin=config.stim_artifact_tmin,
        stim_artifact_tmax=config.stim_artifact_tmax,
        find_flat_channels_meg=config.find_flat_channels_meg,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        mf_reference_run=config.get_mf_reference_run(),
        drop_channels=config.drop_channels,
        find_breaks=config.find_breaks,
        min_break_duration=config.min_break_duration,
        t_break_annot_start_after_previous_event=config.t_break_annot_start_after_previous_event,  # noqa:E501
        t_break_annot_stop_before_next_event=config.t_break_annot_stop_before_next_event,  # noqa:E501
        _raw_split_size=config._raw_split_size,
    )
    return cfg


def main():
    """Run filter."""

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(filter_data)

        # Enabling different runs for different subjects
        sub_run_ses = []
        for subject in config.get_subjects():
            sub_run_ses += list(
                itertools.product(
                    [subject],
                    config.get_runs(subject=subject),
                    config.get_sessions()
                )
            )

        logs = parallel(
            run_func(
                cfg=get_config(subject),
                subject=subject,
                run=run,
                session=session
            ) for subject, run, session in sub_run_ses
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
