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
from typing import Optional, Union
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath, get_bids_path_from_fname

from ..._config_utils import (
    get_sessions, get_runs, get_subjects, get_task, get_datatype,
    get_mf_reference_run,
)
from ..._import_data import (
    import_experimental_data, import_er_data, import_rest_data)
from ..._io import _read_json, _empty_room_match_path
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report, plot_auto_scores_
from ..._run import failsafe_run, save_logs, _update_for_splits
from ..._typing import Literal


def get_input_fnames_frequency_filter(**kwargs):
    """Get paths of files required by filter_data function."""
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    run = kwargs.pop('run')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs

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
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        check=False
    )
    if cfg.use_maxwell_filter:
        path_kwargs['root'] = cfg.deriv_root
        path_kwargs['suffix'] = 'raw'
        path_kwargs['extension'] = '.fif'
        path_kwargs['processing'] = 'sss'
    else:
        path_kwargs['root'] = cfg.bids_root
        path_kwargs['suffix'] = None
        path_kwargs['extension'] = None
        path_kwargs['processing'] = cfg.proc
    bids_path_in = BIDSPath(**path_kwargs)

    in_files = dict()
    in_files[f'raw_run-{run}'] = bids_path_in
    _update_for_splits(in_files, f'raw_run-{run}', single=True)

    if run == cfg.runs[0]:
        do = dict(
            rest=cfg.process_rest and not cfg.task_is_rest,
            noise=cfg.process_empty_room and cfg.datatype == 'meg',
        )
        for task in ('rest', 'noise'):
            if not do[task]:
                continue
            key = f'raw_{task}'
            if cfg.use_maxwell_filter:
                raw_fname = bids_path_in.copy().update(
                    run=None, task=task)
            else:
                if task == 'rest':
                    raw_fname = bids_path_in.copy().update(
                        run=None, task=task)
                else:
                    raw_fname = _read_json(
                        _empty_room_match_path(bids_path_in, cfg))['fname']
                    if raw_fname is not None:
                        raw_fname = get_bids_path_from_fname(raw_fname)
            if raw_fname is None:
                continue
            in_files[key] = raw_fname
            _update_for_splits(
                in_files, key, single=True, allow_missing=True)
            if not in_files[key].fpath.exists():
                in_files.pop(key)

    return in_files


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
    cfg,
    subject: str,
    session: Optional[str],
    run: str,
    in_files: dict,
) -> None:
    """Filter data from a single subject."""
    import matplotlib.pyplot as plt
    out_files = dict()
    bids_path = in_files.pop(f"raw_run-{run}")

    # Create paths for reading and writing the filtered data.
    if cfg.use_maxwell_filter:
        msg = f'Reading: {bids_path.basename}'
        logger.info(**gen_log_kwargs(message=msg))
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

    nice_names = dict(rest='resting-state', noise='empty-room')
    for task in ('rest', 'noise'):
        in_key = f'raw_{task}'
        if in_key not in in_files:
            continue
        data_type = nice_names[task]
        bids_path_noise = in_files.pop(in_key)
        if cfg.use_maxwell_filter:
            msg = (f'Reading {data_type} recording: '
                   f'{bids_path_noise.basename}')
            logger.info(**gen_log_kwargs(message=msg, run=task))
            raw_noise = mne.io.read_raw_fif(bids_path_noise)
        elif data_type == 'empty-room':
            raw_noise = import_er_data(
                cfg=cfg,
                bids_path_er_in=bids_path_noise,
                bids_path_ref_in=bids_path,  # will take bads from this run (0)
            )
        else:
            raw_noise = import_rest_data(
                cfg=cfg,
                bids_path_in=bids_path_noise,
            )
        out_key = f'raw_{task}_filt'
        out_files[out_key] = \
            bids_path.copy().update(
                root=cfg.deriv_root, processing='filt', extension='.fif',
                suffix='raw', split=None, task=task, run=None)

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
            out_files[out_key], overwrite=True, split_naming='bids',
            split_size=cfg._raw_split_size,
        )
        _update_for_splits(out_files, out_key)
        if cfg.interactive:
            # Plot raw data and power spectral density.
            raw_noise.plot(n_channels=50, butterfly=True)
            fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
            raw_noise.plot_psd(fmax=fmax)

    assert len(in_files) == 0, in_files.keys()

    # Report
    with _open_report(cfg=cfg, subject=subject, session=session) as report:
        # This is a weird place for this, but it's the first place we actually
        # start a report, so let's leave it here. We could put it in
        # _import_data (where the auto-bad-finding is called), but that seems
        # worse.
        if run == cfg.runs[0] and cfg.find_noisy_channels_meg:
            msg = 'Adding visualization of noisy channel detection to report.'
            logger.info(**gen_log_kwargs(message=msg))
            figs, captions = plot_auto_scores_(
                cfg=cfg,
                subject=subject,
                session=session,
            )
            tags = ('raw', 'data-quality', *[f'run-{i}' for i in cfg.runs])
            report.add_figure(
                fig=figs,
                caption=captions,
                title='Data Quality',
                tags=tags,
                replace=True,
            )
            for fig in figs:
                plt.close(fig)

        msg = 'Adding filtered raw data to report.'
        logger.info(**gen_log_kwargs(message=msg))
        for fname in out_files.values():
            title = 'Raw'
            if fname.run is not None:
                title += f', run {fname.run}'
            plot_raw_psd = (
                cfg.plot_psd_for_runs == 'all' or
                fname.run in cfg.plot_psd_for_runs
            )
            report.add_raw(
                raw=fname,
                title=title,
                butterfly=5,
                psd=plot_raw_psd,
                tags=('raw', 'filtered', f'run-{fname.run}'),
                # caption=fname.basename,  # TODO upstream
                replace=True,
            )

    return out_files


def get_config(
    *,
    config,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        exec_params=config.exec_params,
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
        config_path=config.config_path,
    )
    if config.sessions == ['N170'] and config.task == 'ERN':
        raise RuntimeError
    return cfg


def main(*, config) -> None:
    """Run filter."""
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            filter_data, exec_params=config.exec_params)

        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                subject=subject,
                session=session,
                run=run,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for run in get_runs(config=config, subject=subject)
        )

    save_logs(config=config, logs=logs)
