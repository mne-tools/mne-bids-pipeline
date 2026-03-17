"""Time-frequency decomposition.

The epoched data is transformed to time-frequency domain using morlet wavelets.
The average power and inter-trial coherence are computed and saved to disk.
"""

from types import SimpleNamespace

import mne
import numpy as np
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_sst,
    _restrict_analyze_channels,
    get_eeg_reference,
    sanitize_cond_name,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _open_report, _sanitize_cond_tag
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_time_frequency(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> InFilesT:
    fname_epochs = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        processing="clean",
        suffix="epo",
        extension=".fif",
        check=False,
    )
    in_files = dict()
    in_files["epochs"] = fname_epochs
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_time_frequency,
)
def run_time_frequency(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    epochs_path = in_files.pop("epochs")
    msg = f"Reading {epochs_path.basename}"
    logger.info(**gen_log_kwargs(message=msg))
    epochs = mne.read_epochs(epochs_path)
    bids_path = epochs_path.copy().update(processing=None, split=None)
    del epochs_path
    _restrict_analyze_channels(epochs, cfg)

    if cfg.time_frequency_subtract_evoked:
        epochs.subtract_evoked()

    freqs = np.arange(cfg.time_frequency_freq_min, cfg.time_frequency_freq_max)

    time_frequency_cycles = cfg.time_frequency_cycles
    if time_frequency_cycles is None:
        time_frequency_cycles = freqs / 3.0

    out_files = dict()
    for condition in cfg.time_frequency_conditions:
        logger.info(**gen_log_kwargs(message=f"Computing TFR for {condition}"))
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True, n_cycles=time_frequency_cycles
        )

        condition_str = sanitize_cond_name(condition)
        power_key = f"power-{condition_str}"
        itc_key = f"itc-{condition_str}"
        out_files[power_key] = bids_path.copy().update(
            suffix=f"power+{condition_str}+tfr", extension=".h5"
        )
        out_files[itc_key] = bids_path.copy().update(
            suffix=f"itc+{condition_str}+tfr", extension=".h5"
        )

        # TODO: verbose='error' here because we write filenames that do not
        # conform to MNE filename checks. This is because BIDS has not
        # finalized how derivatives should be named. Once this is done, we
        # should update our names and/or MNE's checks.
        power.save(out_files[power_key].fpath, overwrite=True, verbose="error")
        itc.save(out_files[itc_key].fpath, overwrite=True, verbose="error")

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        msg = "Adding TFR analysis results to the report."
        logger.info(**gen_log_kwargs(message=msg))
        for condition in cfg.time_frequency_conditions:
            cond = sanitize_cond_name(condition)
            fname_tfr_pow_cond = out_files[f"power-{cond}"].fpath
            fname_tfr_itc_cond = out_files[f"itc-{cond}"].fpath
            with mne.use_log_level("error"):  # filename convention
                power = mne.time_frequency.read_tfrs(fname_tfr_pow_cond, condition=0)
                power.apply_baseline(
                    baseline=cfg.time_frequency_baseline,
                    mode=cfg.time_frequency_baseline_mode,
                )
                if cfg.time_frequency_crop:
                    power.crop(**cfg.time_frequency_crop)
            kwargs = dict(show=False, fig_facecolor="w", font_color="k", border="k")
            fig_power = power.plot_topo(**kwargs)
            report.add_figure(
                fig=fig_power,
                title=f"TFR Power: {condition}",
                caption=f"TFR Power: {condition}",
                tags=("time-frequency", _sanitize_cond_tag(condition)),
                replace=True,
            )
            plt.close(fig_power)
            del power

            with mne.use_log_level("error"):  # filename convention
                itc = mne.time_frequency.read_tfrs(fname_tfr_itc_cond, condition=0)
            fig_itc = itc.plot_topo(**kwargs)
            report.add_figure(
                fig=fig_itc,
                title=f"TFR ITC: {condition}",
                caption=f"TFR Inter-Trial Coherence: {condition}",
                tags=("time-frequency", _sanitize_cond_tag(condition)),
                replace=True,
            )
            plt.close(fig_power)
            del itc

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        time_frequency_conditions=config.time_frequency_conditions,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        time_frequency_cycles=config.time_frequency_cycles,
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked,
        time_frequency_baseline=config.time_frequency_baseline,
        time_frequency_baseline_mode=config.time_frequency_baseline_mode,
        time_frequency_crop=config.time_frequency_crop,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run Time-frequency decomposition."""
    if not config.time_frequency_conditions:
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    sst = _get_sst(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_time_frequency, exec_params=config.exec_params, n_iter=len(sst)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                task=task,
            )
            for subject, session, task in sst
        )
    save_logs(config=config, logs=logs)
