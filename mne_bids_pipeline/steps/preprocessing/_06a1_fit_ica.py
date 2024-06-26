"""Fit ICA.

This fits Independent Component Analysis (ICA) on high-pass filtered raw data,
temporarily creating task-related epochs. The epochs created here are used for
the purpose of fitting ICA only, and will not enter any other processing steps.

Before performing ICA, we reject epochs based on peak-to-peak amplitude above
the 'ica_reject' limits to remove high-amplitude non-biological artifacts
(e.g., voltage or flux spikes).
"""

from types import SimpleNamespace

import autoreject
import mne
import numpy as np
from mne.preprocessing import ICA
from mne_bids import BIDSPath

from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
)
from ..._import_data import annotations_to_events, make_epochs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def get_input_fnames_run_ica(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> dict:
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )
    in_files = dict()
    for run in cfg.runs:
        key = f"raw_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_ica,
)
def run_ica(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: dict,
) -> dict:
    """Run ICA."""
    import matplotlib.pyplot as plt

    if cfg.ica_use_icalabel:
        # The ICALabel network was trained on extended-Infomax ICA decompositions fit
        # on data flltered between 1 and 100 Hz.
        assert cfg.ica_algorithm in ["picard-extended_infomax", "extended_infomax"]
        assert cfg.ica_l_freq == 1.0
        assert cfg.h_freq == 100.0
        assert cfg.eeg_reference == "average"

    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs]
    out_files = dict()
    bids_basename = raw_fnames[0].copy().update(processing=None, split=None, run=None)
    out_files["ica"] = bids_basename.copy().update(processing="icafit", suffix="ica")
    out_files["epochs"] = (
        out_files["ica"].copy().update(suffix="epo", processing="icafit")
    )
    del bids_basename

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.

    # Generate a unique event name -> event code mapping that can be used
    # across all runs.
    event_name_to_code_map = annotations_to_events(raw_paths=raw_fnames)

    epochs = None
    for idx, (run, raw_fname) in enumerate(zip(cfg.runs, raw_fnames)):
        msg = f"Processing raw data from {raw_fname.basename}"
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        # Produce high-pass filtered version of the data for ICA.
        # Sanity check – make sure we're using the correct data!
        if cfg.raw_resample_sfreq is not None:
            assert np.allclose(raw.info["sfreq"], cfg.raw_resample_sfreq)
        if cfg.l_freq is not None:
            assert np.allclose(raw.info["highpass"], cfg.l_freq)

        if idx == 0:
            if cfg.ica_l_freq is None:
                msg = (
                    f"Not applying high-pass filter (data is already filtered, "
                    f'cutoff: {raw.info["highpass"]} Hz).'
                )
                logger.info(**gen_log_kwargs(message=msg))
            else:
                msg = f"Applying high-pass filter with {cfg.ica_l_freq} Hz cutoff …"
                logger.info(**gen_log_kwargs(message=msg))
                raw.filter(l_freq=cfg.ica_l_freq, h_freq=None, n_jobs=1)

        # Only keep the subset of the mapping that applies to the current run
        event_id = event_name_to_code_map.copy()
        for event_name in event_id.copy().keys():
            if event_name not in raw.annotations.description:
                del event_id[event_name]

        if idx == 0:
            msg = "Creating task-related epochs …"
            logger.info(**gen_log_kwargs(message=msg))
        these_epochs = make_epochs(
            subject=subject,
            session=session,
            task=cfg.task,
            conditions=cfg.conditions,
            raw=raw,
            event_id=event_id,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            metadata_tmin=cfg.epochs_metadata_tmin,
            metadata_tmax=cfg.epochs_metadata_tmax,
            metadata_keep_first=cfg.epochs_metadata_keep_first,
            metadata_keep_last=cfg.epochs_metadata_keep_last,
            metadata_query=cfg.epochs_metadata_query,
            event_repeated=cfg.event_repeated,
            epochs_decim=cfg.epochs_decim,
            task_is_rest=cfg.task_is_rest,
            rest_epochs_duration=cfg.rest_epochs_duration,
            rest_epochs_overlap=cfg.rest_epochs_overlap,
        )

        these_epochs.load_data()  # Remove reference to raw
        del raw  # free memory

        if epochs is None:
            epochs = these_epochs
        else:
            epochs = mne.concatenate_epochs([epochs, these_epochs], on_mismatch="warn")

        del these_epochs
    del run

    # Set an EEG reference
    if "eeg" in cfg.ch_types:
        if cfg.ica_use_icalabel:
            assert cfg.eeg_reference == "average"
            projection = False  # Avg. ref. needs to be applied for MNE-ICALabel
        elif cfg.eeg_reference == "average":
            projection = True
        else:
            projection = False

        if not projection:
            msg = "Applying average reference to EEG epochs used for ICA fitting."
            logger.info(**gen_log_kwargs(message=msg))

        epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

    ar_reject_log = ar_n_interpolate_ = None
    if cfg.ica_reject == "autoreject_local":
        msg = (
            "Using autoreject to find bad epochs for ICA "
            "(no interpolation will be performend)"
        )
        logger.info(**gen_log_kwargs(message=msg))
        ar = autoreject.AutoReject(
            n_interpolate=cfg.autoreject_n_interpolate,
            random_state=cfg.random_state,
            n_jobs=exec_params.n_jobs,
            verbose=False,
        )
        ar.fit(epochs)
        ar_reject_log = ar.get_reject_log(epochs)
        epochs = epochs[~ar_reject_log.bad_epochs]

        n_epochs_before_reject = len(epochs)
        n_epochs_rejected = ar_reject_log.bad_epochs.sum()
        n_epochs_after_reject = n_epochs_before_reject - n_epochs_rejected

        ar_n_interpolate_ = ar.n_interpolate_
        msg = (
            f"autoreject marked {n_epochs_rejected} epochs as bad "
            f"(cross-validated n_interpolate limit: {ar_n_interpolate_})"
        )
        logger.info(**gen_log_kwargs(message=msg))
        del ar
    else:
        # Reject epochs based on peak-to-peak rejection thresholds
        ica_reject = _get_reject(
            subject=subject,
            session=session,
            reject=cfg.ica_reject,
            ch_types=cfg.ch_types,
            param="ica_reject",
        )
        n_epochs_before_reject = len(epochs)
        epochs.drop_bad(reject=ica_reject)
        n_epochs_after_reject = len(epochs)
        n_epochs_rejected = n_epochs_before_reject - n_epochs_after_reject

        msg = (
            f"Removed {n_epochs_rejected} of {n_epochs_before_reject} epochs via PTP "
            f"rejection thresholds: {ica_reject}"
        )
        logger.info(**gen_log_kwargs(message=msg))
        ar = None

    if 0 < n_epochs_after_reject < 0.5 * n_epochs_before_reject:
        msg = (
            "More than 50% of all epochs rejected. Please check the "
            "rejection thresholds."
        )
        logger.warning(**gen_log_kwargs(message=msg))
    elif n_epochs_after_reject == 0:
        rejection_type = (
            cfg.ica_reject if cfg.ica_reject == "autoreject_local" else "PTP-based"
        )
        raise RuntimeError(
            f"No epochs remaining after {rejection_type} rejection. Cannot continue."
        )

    msg = f"Saving {n_epochs_after_reject} ICA epochs to disk."
    logger.info(**gen_log_kwargs(message=msg))
    epochs.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")

    msg = f"Calculating ICA solution using method: {cfg.ica_algorithm}."
    logger.info(**gen_log_kwargs(message=msg))

    algorithm = cfg.ica_algorithm
    fit_params = None

    if algorithm == "picard":
        fit_params = dict(fastica_it=5)
    elif algorithm == "picard-extended_infomax":
        algorithm = "picard"
        fit_params = dict(ortho=False, extended=True)
    elif algorithm == "extended_infomax":
        algorithm = "infomax"
        fit_params = dict(extended=True)

    ica = ICA(
        method=algorithm,
        random_state=cfg.random_state,
        n_components=cfg.ica_n_components,
        fit_params=fit_params,
        max_iter=cfg.ica_max_iterations,
    )
    ica.fit(epochs, decim=cfg.ica_decim)
    explained_var = (
        ica.pca_explained_variance_[: ica.n_components_].sum()
        / ica.pca_explained_variance_.sum()
    )
    msg = (
        f"Fit {ica.n_components_} components (explaining "
        f"{round(explained_var * 100, 1)}% of the variance) in "
        f"{ica.n_iter_} iterations."
    )
    logger.info(**gen_log_kwargs(message=msg))
    msg = "Saving ICA solution to disk."
    logger.info(**gen_log_kwargs(message=msg))
    ica.save(out_files["ica"], overwrite=True)

    # Add to report
    tags = ("ica", "epochs")
    title = "ICA: epochs for fitting"
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        task=cfg.task,
    ) as report:
        report.add_epochs(
            epochs=epochs,
            title=title,
            drop_log_ignore=(),
            replace=True,
            tags=tags,
        )
        if cfg.ica_reject == "autoreject_local":
            caption = (
                f"Autoreject was run to produce cleaner epochs before fitting ICA. "
                f"{ar_reject_log.bad_epochs.sum()} epochs were rejected because more "
                f"than {ar_n_interpolate_} channels were bad (cross-validated "
                f"n_interpolate limit; excluding globally bad and non-data channels, "
                f"shown in white). Note that none of the blue segments were actually "
                f"interpolated before submitting the data to ICA. This is following "
                f"the recommended approach for ICA described in the the Autoreject "
                f"documentation."
            )
            fig = ar_reject_log.plot(
                orientation="horizontal", aspect="auto", show=False
            )
            report.add_figure(
                fig=fig,
                title="Autoreject cleaning",
                section=title,
                caption=caption,
                tags=tags + ("autoreject",),
                replace=True,
            )
            plt.close(fig)
            del caption
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        conditions=config.conditions,
        runs=get_runs(config=config, subject=subject),
        task_is_rest=config.task_is_rest,
        ica_l_freq=config.ica_l_freq,
        ica_algorithm=config.ica_algorithm,
        ica_n_components=config.ica_n_components,
        ica_max_iterations=config.ica_max_iterations,
        ica_decim=config.ica_decim,
        ica_reject=config.ica_reject,
        ica_use_icalabel=config.ica_use_icalabel,
        autoreject_n_interpolate=config.autoreject_n_interpolate,
        random_state=config.random_state,
        ch_types=config.ch_types,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        epochs_decim=config.epochs_decim,
        raw_resample_sfreq=config.raw_resample_sfreq,
        event_repeated=config.event_repeated,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        epochs_metadata_tmin=config.epochs_metadata_tmin,
        epochs_metadata_tmax=config.epochs_metadata_tmax,
        epochs_metadata_keep_first=config.epochs_metadata_keep_first,
        epochs_metadata_keep_last=config.epochs_metadata_keep_last,
        epochs_metadata_query=config.epochs_metadata_query,
        eeg_reference=get_eeg_reference(config),
        eog_channels=config.eog_channels,
        rest_epochs_duration=config.rest_epochs_duration,
        rest_epochs_overlap=config.rest_epochs_overlap,
        processing="filt" if config.regress_artifact is None else "regress",
        _epochs_split_size=config._epochs_split_size,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run ICA."""
    if config.spatial_filter != "ica":
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(run_ica, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
