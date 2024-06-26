"""Find ICA artifacts.

This step automatically finds ECG- and EOG-related ICs in your data, and sets them
as bad components.

To actually remove designated ICA components from your data, you will have to
run the apply_ica step.
"""

from types import SimpleNamespace
from typing import Literal

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.viz import plot_ica_components
from mne_bids import BIDSPath
from mne_icalabel import label_components
import mne_icalabel
import matplotlib.pyplot as plt

from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def detect_bad_components(
    *,
    cfg,
    which: Literal["eog", "ecg"],
    epochs: mne.BaseEpochs | None,
    ica: mne.preprocessing.ICA,
    ch_names: list[str] | None,
    subject: str,
    session: str | None,
) -> tuple[list[int], np.ndarray]:
    artifact = which.upper()
    if epochs is None:
        msg = (
            f"No {artifact} events could be found. "
            f"Not running {artifact} artifact detection."
        )
        logger.info(**gen_log_kwargs(message=msg))
        return [], []
    msg = f"Performing automated {artifact} artifact detection …"
    logger.info(**gen_log_kwargs(message=msg))

    if which == "eog":
        inds, scores = ica.find_bads_eog(
            epochs,
            threshold=cfg.ica_eog_threshold,
            ch_name=ch_names,
        )
    else:
        inds, scores = ica.find_bads_ecg(
            epochs,
            method="ctps",
            threshold=cfg.ica_ecg_threshold,
            ch_name=ch_names,
        )

    if not inds:
        adjust_setting = f"ica_{which}_threshold"
        warn = (
            f"No {artifact}-related ICs detected, this is highly "
            f"suspicious. A manual check is suggested. You may wish to "
            f'lower "{adjust_setting}".'
        )
        logger.warning(**gen_log_kwargs(message=warn))
    else:
        msg = (
            f"Detected {len(inds)} {artifact}-related ICs in "
            f"{len(epochs)} {artifact} epochs: {', '.join([str(i) for i in inds])}"
        )
        logger.info(**gen_log_kwargs(message=msg))

    return inds, scores


def get_input_fnames_find_ica_artifacts(
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
    in_files["epochs"] = bids_basename.copy().update(processing="icafit", suffix="epo")
    _update_for_splits(in_files, "epochs", single=True)
    for run in cfg.runs:
        key = f"raw_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)
    in_files["ica"] = bids_basename.copy().update(processing="icafit", suffix="ica")
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_find_ica_artifacts,
)
def find_ica_artifacts(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: dict,
) -> dict:
    """Run ICA."""
    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs]
    bids_basename = raw_fnames[0].copy().update(processing=None, split=None, run=None)
    out_files = dict()
    out_files["ica"] = bids_basename.copy().update(processing="ica", suffix="ica")
    out_files["ecg"] = bids_basename.copy().update(processing="ica+ecg", suffix="ave")
    out_files["eog"] = bids_basename.copy().update(processing="ica+eog", suffix="ave")

    # DO NOT add this to out_files["ica"] because we expect it to be modified by users.
    # If the modify it and it's in out_files, caching will detect the hash change and
    # consider *this step* a cache miss, and it will run again, overwriting the user's
    # changes. Instead, we want the ica.apply step to rerun (which it will if the
    # file changes).
    out_files_components = bids_basename.copy().update(
        processing="ica", suffix="components", extension=".tsv"
    )
    del bids_basename
    msg = "Loading ICA solution"
    logger.info(**gen_log_kwargs(message=msg))
    ica = mne.preprocessing.read_ica(in_files.pop("ica"))

    # Epochs used for ICA fitting
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    # ECG component detection
    epochs_ecg = None
    ecg_ics, ecg_scores = [], []
    if cfg.ica_use_ecg_detection:
        for ri, raw_fname in enumerate(raw_fnames):
            # Have the channels needed to make ECG epochs
            raw = mne.io.read_raw(raw_fname, preload=False)
            # ECG epochs
            if not (
                "ecg" in raw.get_channel_types()
                or "meg" in cfg.ch_types
                or "mag" in cfg.ch_types
            ):
                msg = (
                    "No ECG or magnetometer channels are present, cannot "
                    "automate artifact detection for ECG."
                )
                logger.info(**gen_log_kwargs(message=msg))
                break
            elif ri == 0:
                msg = "Creating ECG epochs …"
                logger.info(**gen_log_kwargs(message=msg))

            # We want to extract a total of 5 min of data for ECG epochs generation
            # (across all runs)
            total_ecg_dur = 5 * 60
            ecg_dur_per_run = total_ecg_dur / len(raw_fnames)
            t_mid = (raw.times[-1] + raw.times[0]) / 2
            raw = raw.crop(
                tmin=max(t_mid - 1 / 2 * ecg_dur_per_run, 0),
                tmax=min(t_mid + 1 / 2 * ecg_dur_per_run, raw.times[-1]),
            ).load_data()

            these_ecg_epochs = create_ecg_epochs(
                raw,
                baseline=(None, -0.2),
                tmin=-0.5,
                tmax=0.5,
            )
            del raw  # Free memory
            if len(these_ecg_epochs):
                if epochs.reject is not None:
                    these_ecg_epochs.drop_bad(reject=epochs.reject)
                if len(these_ecg_epochs):
                    if epochs_ecg is None:
                        epochs_ecg = these_ecg_epochs
                    else:
                        epochs_ecg = mne.concatenate_epochs(
                            [epochs_ecg, these_ecg_epochs], on_mismatch="warn"
                        )
            del these_ecg_epochs
        else:  # did not break so had usable channels
            ecg_ics, ecg_scores = detect_bad_components(
                cfg=cfg,
                which="ecg",
                epochs=epochs_ecg,
                ica=ica,
                ch_names=None,  # we currently don't allow for custom channels
                subject=subject,
                session=session,
            )

    # EOG component detection
    epochs_eog = None
    eog_ics = eog_scores = []
    if cfg.ica_use_eog_detection:
        for ri, raw_fname in enumerate(raw_fnames):
            raw = mne.io.read_raw_fif(raw_fname, preload=True)
            if cfg.eog_channels:
                ch_names = cfg.eog_channels
                assert all([ch_name in raw.ch_names for ch_name in ch_names])
            else:
                eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
                ch_names = [raw.ch_names[pick] for pick in eog_picks]
            if not ch_names:
                msg = "No EOG channel is present, cannot automate IC detection for EOG."
                logger.info(**gen_log_kwargs(message=msg))
                break
            elif ri == 0:
                msg = "Creating EOG epochs …"
                logger.info(**gen_log_kwargs(message=msg))
            these_eog_epochs = create_eog_epochs(
                raw,
                ch_name=ch_names,
                baseline=(None, -0.2),
            )
            if len(these_eog_epochs):
                if epochs.reject is not None:
                    these_eog_epochs.drop_bad(reject=epochs.reject)
                if len(these_eog_epochs):
                    if epochs_eog is None:
                        epochs_eog = these_eog_epochs
                    else:
                        epochs_eog = mne.concatenate_epochs(
                            [epochs_eog, these_eog_epochs], on_mismatch="warn"
                        )
        else:  # did not break
            eog_ics, eog_scores = detect_bad_components(
                cfg=cfg,
                which="eog",
                epochs=epochs_eog,
                ica=ica,
                ch_names=cfg.eog_channels,
                subject=subject,
                session=session,
            )
    

    # Run MNE-ICALabel if requested.
    if cfg.ica_use_icalabel:
        icalabel_ics = []
        icalabel_labels = []
        icalabel_prob = []
        msg = "Performing automated artifact detection (MNE-ICALabel) …"
        logger.info(**gen_log_kwargs(message=msg))
        
        label_results = mne_icalabel.label_components(inst=epochs, ica=ica, method="iclabel")
        for idx, (label,prob) in enumerate(zip(label_results["labels"],label_results["y_pred_proba"])):
            #icalabel_include = ["brain", "other"]
            print(label)
            print(prob)

            if label not in cfg.icalabel_include:
                icalabel_ics.append(idx)
                icalabel_labels.append(label)
                icalabel_prob.append(prob)

        msg = (
            f"Detected {len(icalabel_ics)} artifact-related independent component(s) "
            f"in {len(epochs)} epochs."
        )
        logger.info(**gen_log_kwargs(message=msg))
    else:
        icalabel_ics = []

    ica.exclude = sorted(set(ecg_ics + eog_ics + icalabel_ics))

    # Save updated ICA to disk.
    # We also store the automatically identified ECG- and EOG-related ICs.
    msg = "Saving ICA solution and detected artifacts to disk."
    logger.info(**gen_log_kwargs(message=msg))
    ica.save(out_files["ica"], overwrite=True)

    # Create TSV.
    tsv_data = pd.DataFrame(
        dict(
            component=list(range(ica.n_components_)),
            type=["ica"] * ica.n_components_,
            description=["Independent Component"] * ica.n_components_,
            status=["good"] * ica.n_components_,
            status_description=["n/a"] * ica.n_components_,
        )
    )

    if cfg.ica_use_icalabel:
        assert len(icalabel_ics) == len(icalabel_labels)
        for component, label in zip(icalabel_ics, icalabel_labels):
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[
                row_idx, "status_description"
            ] = f"Auto-detected {label} (MNE-ICALabel)"
    if cfg.ica_use_ecg_detection:
        for component in ecg_ics:
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[
                row_idx, "status_description"
            ] = "Auto-detected ECG artifact (MNE)"
    if cfg.ica_use_eog_detection:
        for component in eog_ics:
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[
                row_idx, "status_description"
            ] = "Auto-detected EOG artifact (MNE)"

    tsv_data.to_csv(out_files_components, sep="\t", index=False)

    # Lastly, add info about the epochs used for the ICA fit, and plot all ICs
    # for manual inspection.

    ecg_evoked = None if epochs_ecg is None else epochs_ecg.average()
    eog_evoked = None if epochs_eog is None else epochs_eog.average()

    
    ecg_scores = None if len(ecg_scores) == 0 else ecg_scores
    eog_scores = None if len(eog_scores) == 0 else eog_scores

    # Save ECG and EOG evokeds to disk.
    for artifact_name, artifact_evoked in zip(("ecg", "eog"), (ecg_evoked, eog_evoked)):
        if artifact_evoked:
            msg = f"Saving {artifact_name.upper()} artifact: {out_files[artifact_name]}"
            logger.info(**gen_log_kwargs(message=msg))
            artifact_evoked.save(out_files[artifact_name], overwrite=True)
        else:
            # Don't track the non-existent output file
            del out_files[artifact_name]

    del artifact_name, artifact_evoked

    title = "ICA: components"
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        task=cfg.task,
    ) as report:
        logger.info(**gen_log_kwargs(message=f'Adding "{title}" to report.'))
        report.add_ica(
            ica=ica,
            title=title,
            inst=epochs,
            ecg_evoked=ecg_evoked,
            eog_evoked=eog_evoked,
            ecg_scores=ecg_scores,
            eog_scores=eog_scores,
            replace=True,
            n_jobs=1,  # avoid automatic parallelization
            tags=("ica",),  # the default but be explicit
        )

        # Add a plot for each excluded IC together with the given label and the probability
        # TODO: Improve this plot e.g. combine all figures in one plot
        for ic, label, prob in zip(icalabel_ics, icalabel_labels, icalabel_prob):
            excluded_IC_figure = plot_ica_components(
                ica=ica,
                picks=ic,
            )
            excluded_IC_figure.axes[0].text(0, -0.15, f"Label: {label} \n Probability: {prob:.3f}", ha="center", fontsize=8, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

            report.add_figure(
                fig=excluded_IC_figure,
                title = f'ICA{ic:03}',
                replace=True,
            )
            plt.close(excluded_IC_figure)

    msg = 'Carefully review the extracted ICs and mark components "bad" in:'
    logger.info(**gen_log_kwargs(message=msg, emoji="🛑"))
    logger.info(**gen_log_kwargs(message=str(out_files_components), emoji="🛑"))

    assert len(in_files) == 0, in_files.keys()
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
        ica_reject=config.ica_reject,
        ica_use_eog_detection = config.ica_use_eog_detection,
        ica_eog_threshold=config.ica_eog_threshold,
        ica_use_ecg_detection = config.ica_use_ecg_detection,
        ica_ecg_threshold=config.ica_ecg_threshold,
        ica_use_icalabel=config.ica_use_icalabel,
        icalabel_include = config.icalabel_include,
        autoreject_n_interpolate=config.autoreject_n_interpolate,
        random_state=config.random_state,
        ch_types=config.ch_types,
        l_freq=config.l_freq,
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
        parallel, run_func = parallel_func(
            find_ica_artifacts, exec_params=config.exec_params
        )
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
