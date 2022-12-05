"""Apply low- and high-pass filters.

The data are bandpass filtered to the frequencies defined in the config
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

import numpy as np
from types import SimpleNamespace
from typing import Optional, Union, Literal

import mne

from ..._config_utils import (
    get_sessions, get_runs, get_subjects, get_task, get_datatype,
    get_mf_reference_run,
)
from ..._import_data import (
    import_experimental_data, import_er_data, _get_raw_paths,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report, _add_raw
from ..._run import failsafe_run, save_logs, _update_for_splits


def get_input_fnames_frequency_filter(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: str,
) -> dict:
    """Get paths of files required by filter_data function."""
    kind = 'sss' if cfg.use_maxwell_filter else 'orig'
    return _get_raw_paths(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        kind=kind,
        include_mf_ref=False,
    )


def filter(
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: str,
    l_freq: Optional[float],
    h_freq: Optional[float],
    l_trans_bandwidth: Union[float, Literal['auto']],
    h_trans_bandwidth: Union[float, Literal['auto']],
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

    logger.info(**gen_log_kwargs(message=msg))

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
    run: str,
    sfreq: float,
    data_type: Literal['experimental', 'empty-room', 'resting-state']
) -> None:
    if not sfreq:
        return

    msg = f'Resampling {data_type} data to {sfreq:.1f} Hz'
    logger.info(**gen_log_kwargs(message=msg))
    raw.resample(sfreq, npad='auto')


@failsafe_run(
    get_input_fnames=get_input_fnames_frequency_filter,
)
def filter_data(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: str,
    in_files: dict,
) -> dict:
    """Filter data from a single subject."""
    out_files = dict()
    in_key = f"raw_run-{run}"
    bids_path = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f'{in_key}-bads', None)

    # Create paths for reading and writing the filtered data.
    if cfg.use_maxwell_filter:
        msg = f'Reading: {bids_path.basename}'
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(bids_path)
    else:
        raw = import_experimental_data(
            cfg=cfg,
            bids_path_in=bids_path,
            bids_path_bads_in=bids_path_bads_in,
            data_is_rest=False,
        )

    out_files[in_key] = bids_path.copy().update(
        root=cfg.deriv_root, processing='filt', extension='.fif',
        suffix='raw', split=None)
    raw.load_data()
    filter(
        raw=raw, subject=subject, session=session, run=run,
        h_freq=cfg.h_freq, l_freq=cfg.l_freq,
        h_trans_bandwidth=cfg.h_trans_bandwidth,
        l_trans_bandwidth=cfg.l_trans_bandwidth,
        data_type='experimental',
    )
    resample(raw=raw, subject=subject, session=session, run=run,
             sfreq=cfg.raw_resample_sfreq, data_type='experimental')

    raw.save(out_files[in_key], overwrite=True, split_naming='bids',
             split_size=cfg._raw_split_size)
    _update_for_splits(out_files, in_key)
    if exec_params.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
        raw.plot_psd(fmax=fmax)

    del raw

    nice_names = dict(rest='resting-state', noise='empty-room')
    for task in ('rest', 'noise'):
        in_key = f'raw_{task}'
        if in_key not in in_files:
            continue
        data_type = nice_names[task]
        bids_path_noise = in_files.pop(in_key)
        bids_path_noise_bads = in_files.pop(f'{in_key}-bads', None)
        if cfg.use_maxwell_filter:
            msg = (f'Reading {data_type} recording: '
                   f'{bids_path_noise.basename}')
            logger.info(**gen_log_kwargs(message=msg, run=task))
            raw_noise = mne.io.read_raw_fif(bids_path_noise)
        elif data_type == 'empty-room':
            raw_noise = import_er_data(
                cfg=cfg,
                bids_path_er_in=bids_path_noise,
                bids_path_ref_in=bids_path,
                bids_path_er_bads_in=bids_path_noise_bads,
                # take bads from this run (0)
                bids_path_ref_bads_in=bids_path_bads_in,
                prepare_maxwell_filter=False,
            )
        else:
            raw_noise = import_experimental_data(
                cfg=cfg,
                bids_path_in=bids_path_noise,
                bids_path_bads_in=bids_path_noise_bads,
                data_is_rest=True,
            )
        out_files[in_key] = \
            bids_path.copy().update(
                root=cfg.deriv_root, processing='filt', extension='.fif',
                suffix='raw', split=None, task=task, run=None)

        raw_noise.load_data()
        filter(
            raw=raw_noise, subject=subject, session=session, run=task,
            h_freq=cfg.h_freq, l_freq=cfg.l_freq,
            h_trans_bandwidth=cfg.h_trans_bandwidth,
            l_trans_bandwidth=cfg.l_trans_bandwidth,
            data_type=data_type,
        )
        resample(raw=raw_noise, subject=subject, session=session, run=task,
                 sfreq=cfg.raw_resample_sfreq, data_type=data_type)

        raw_noise.save(
            out_files[in_key], overwrite=True, split_naming='bids',
            split_size=cfg._raw_split_size,
        )
        _update_for_splits(out_files, in_key)
        if exec_params.interactive:
            # Plot raw data and power spectral density.
            raw_noise.plot(n_channels=50, butterfly=True)
            fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
            raw_noise.plot_psd(fmax=fmax)

    assert len(in_files) == 0, in_files.keys()

    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session,
            run=run) as report:
        msg = 'Adding filtered raw data to report'
        logger.info(**gen_log_kwargs(message=msg))
        for fname in out_files.values():
            _add_raw(
                cfg=cfg,
                report=report,
                bids_path_in=fname,
                title='Raw (filtered)',
            )

    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        reader_extra_params=config.reader_extra_params,
        process_empty_room=config.process_empty_room,
        process_rest=config.process_rest,
        task_is_rest=config.task_is_rest,
        runs=get_runs(config=config, subject=subject),
        use_maxwell_filter=config.use_maxwell_filter,
        proc=config.proc,
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.bids_root,
        deriv_root=config.deriv_root,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        l_trans_bandwidth=config.l_trans_bandwidth,
        h_trans_bandwidth=config.h_trans_bandwidth,
        raw_resample_sfreq=config.raw_resample_sfreq,
        crop_runs=config.crop_runs,
        rename_events=config.rename_events,
        eeg_bipolar_channels=config.eeg_bipolar_channels,
        eeg_template_montage=config.eeg_template_montage,
        fix_stim_artifact=config.fix_stim_artifact,
        stim_artifact_tmin=config.stim_artifact_tmin,
        stim_artifact_tmax=config.stim_artifact_tmax,
        find_flat_channels_meg=config.find_flat_channels_meg,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        mf_reference_run=get_mf_reference_run(config),
        drop_channels=config.drop_channels,
        find_breaks=config.find_breaks,
        min_break_duration=config.min_break_duration,
        t_break_annot_start_after_previous_event=config.t_break_annot_start_after_previous_event,  # noqa:E501
        t_break_annot_stop_before_next_event=config.t_break_annot_stop_before_next_event,  # noqa:E501
        data_type=config.data_type,
        ch_types=config.ch_types,
        eog_channels=config.eog_channels,
        on_rename_missing_events=config.on_rename_missing_events,
        plot_psd_for_runs=config.plot_psd_for_runs,
        _raw_split_size=config._raw_split_size,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run filter."""
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            filter_data, exec_params=config.exec_params)

        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for run in get_runs(config=config, subject=subject)
        )

    save_logs(config=config, logs=logs)
