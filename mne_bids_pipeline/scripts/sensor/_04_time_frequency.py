"""Time-frequency decomposition.

The epoched data is transformed to time-frequency domain using morlet wavelets.
The average power and inter-trial coherence are computed and saved to disk.
"""

import itertools
from types import SimpleNamespace

import numpy as np

import mne

from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_deriv_root,
    _restrict_analyze_channels, get_eeg_reference, sanitize_cond_name,
)
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, save_logs, _script_path
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _open_report, _sanitize_cond_tag


def get_input_fnames_time_frequency(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    processing = None
    if cfg.spatial_filter is not None:
        processing = 'clean'
    fname_epochs = BIDSPath(subject=subject,
                            session=session,
                            task=cfg.task,
                            acquisition=cfg.acq,
                            run=None,
                            recording=cfg.rec,
                            space=cfg.space,
                            datatype=cfg.datatype,
                            root=cfg.deriv_root,
                            processing=processing,
                            suffix='epo',
                            extension='.fif',
                            check=False)
    in_files = dict()
    in_files['epochs'] = fname_epochs
    return in_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_time_frequency)
def run_time_frequency(*, cfg, subject, session, in_files):
    import matplotlib.pyplot as plt
    msg = f'Input: {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    bids_path = in_files['epochs'].copy().update(processing=None)

    epochs = mne.read_epochs(in_files.pop('epochs'))
    _restrict_analyze_channels(epochs, cfg)

    if cfg.time_frequency_subtract_evoked:
        epochs.subtract_evoked()

    freqs = np.arange(cfg.time_frequency_freq_min,
                      cfg.time_frequency_freq_max)

    time_frequency_cycles = cfg.time_frequency_cycles
    if time_frequency_cycles is None:
        time_frequency_cycles = freqs / 3.

    out_files = dict()
    for condition in cfg.time_frequency_conditions:
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True,
            n_cycles=time_frequency_cycles
        )

        condition_str = sanitize_cond_name(condition)
        power_key = f'power-{condition_str}'
        itc_key = f'itc-{condition_str}'
        out_files[power_key] = bids_path.copy().update(
            suffix=f'power+{condition_str}+tfr', extension='.h5')
        out_files[itc_key] = bids_path.copy().update(
            suffix=f'itc+{condition_str}+tfr', extension='.h5')

        # TODO: verbose='error' here because we write filenames that do not
        # conform to MNE filename checks. This is because BIDS has not
        # finalized how derivatives should be named. Once this is done, we
        # should update our names and/or MNE's checks.
        power.save(out_files[power_key], overwrite=True, verbose='error')
        itc.save(out_files[itc_key], overwrite=True, verbose='error')

    # Report
    with _open_report(cfg=cfg, subject=subject, session=session) as report:
        msg = 'Adding TFR analysis results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )
        for condition in cfg.time_frequency_conditions:
            cond = sanitize_cond_name(condition)
            fname_tfr_pow_cond = out_files[f'power-{cond}']
            fname_tfr_itc_cond = out_files[f'itc-{cond}']
            with mne.use_log_level('error'):  # filename convention
                power = mne.time_frequency.read_tfrs(
                    fname_tfr_pow_cond, condition=0)
                power.apply_baseline(
                    baseline=cfg.time_frequency_baseline,
                    mode=cfg.time_frequency_baseline_mode)
                if cfg.time_frequency_crop:
                    power.crop(**cfg.time_frequency_crop)
            kwargs = dict(
                show=False, fig_facecolor='w', font_color='k', border='k'
            )
            fig_power = power.plot_topo(**kwargs)
            report.add_figure(
                fig=fig_power,
                title=f'TFR Power: {condition}',
                caption=f'TFR Power: {condition}',
                tags=('time-frequency', _sanitize_cond_tag(condition)),
                replace=True,
            )
            plt.close(fig_power)
            del power

            with mne.use_log_level('error'):  # filename convention
                itc = mne.time_frequency.read_tfrs(
                    fname_tfr_itc_cond, condition=0)
            fig_itc = itc.plot_topo(**kwargs)
            report.add_figure(
                fig=fig_itc,
                title=f'TFR ITC: {condition}',
                caption=f'TFR Inter-Trial Coherence: {condition}',
                tags=('time-frequency', _sanitize_cond_tag(condition)),
                replace=True,
            )
            plt.close(fig_power)
            del itc

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=get_deriv_root(config),
        time_frequency_conditions=config.time_frequency_conditions,
        analyze_channels=config.analyze_channels,
        spatial_filter=config.spatial_filter,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        time_frequency_cycles=config.time_frequency_cycles,
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked,
        time_frequency_baseline=config.time_frequency_baseline,
        time_frequency_baseline_mode=config.time_frequency_baseline_mode,
        time_frequency_crop=config.time_frequency_crop,
    )
    return cfg


def main(*, config) -> None:
    """Run Time-frequency decomposition."""
    if not config.time_frequency_conditions:
        msg = 'Skipping …'
        with _script_path(__file__):
            logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func = parallel_func(run_time_frequency, config=config)
    with get_parallel_backend(config):
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                subject=subject,
                session=session,
            )
            for subject, session in
            itertools.product(get_subjects(config),
                              get_sessions(config))
        )
    save_logs(config=config, logs=logs)
