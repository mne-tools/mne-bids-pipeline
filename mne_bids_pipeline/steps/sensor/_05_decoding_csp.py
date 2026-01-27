"""Decoding based on common spatial patterns (CSP)."""

import os.path as op
from types import SimpleNamespace

import matplotlib.transforms
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_decoding_proc,
    _get_ss,
    _restrict_analyze_channels,
    get_decoding_contrasts,
    get_eeg_reference,
)
from mne_bids_pipeline._decoding import (
    LogReg,
    _decoding_preproc_steps,
    _handle_csp_args,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import (
    _imshow_tf,
    _open_report,
    _plot_full_epochs_decoding_scores,
    _sanitize_cond_tag,
)
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, IntArrayT, OutFilesT


def _prepare_labels(*, epochs: mne.BaseEpochs, contrast: tuple[str, str]) -> IntArrayT:
    """Return the projection of the events_id on a boolean vector.

    This projection is useful in the case of hierarchical events:
    we project the different events contained in one condition into
    just one label.

    Returns
    -------
    A boolean numpy array containing the labels.
    """
    epochs_cond_0 = epochs[contrast[0]]
    event_codes_condition_0 = set(epochs_cond_0.events[:, 2])
    epochs_cond_1 = epochs[contrast[1]]
    event_codes_condition_1 = set(epochs_cond_1.events[:, 2])

    y: IntArrayT = epochs.events[:, 2].copy()
    for i in range(len(y)):
        if y[i] in event_codes_condition_0 and y[i] in event_codes_condition_1:
            msg = (
                f"Event_id {y[i]} is contained both in "
                f"{contrast[0]}'s set {event_codes_condition_0} and in "
                f"{contrast[1]}'s set {event_codes_condition_1}."
                f"{contrast} does not constitute a valid partition."
            )
            raise RuntimeError(msg)
        elif y[i] in event_codes_condition_0:
            y[i] = 0
        elif y[i] in event_codes_condition_1:
            y[i] = 1
        else:
            # This should not happen because epochs should already be filtered
            msg = (
                f"Event_id {y[i]} is not contained in "
                f"{contrast[0]}'s set {event_codes_condition_0}  nor in "
                f"{contrast[1]}'s set {event_codes_condition_1}."
            )
            raise RuntimeError(msg)
    return y


def prepare_epochs_and_y(
    *,
    epochs: mne.BaseEpochs,
    contrast: tuple[str, str],
    cfg: SimpleNamespace,
    fmin: float,
    fmax: float,
) -> tuple[mne.BaseEpochs, IntArrayT]:
    """Band-pass between, sub-select the desired epochs, and prepare y."""
    # filtering out the conditions we are not interested in, to ensure here we
    # have a valid partition between the condition of the contrast.

    # XXX Hack for handling epochs selection via metadata
    # This also makes a copy
    if contrast[0].startswith("event_name.isin"):
        epochs_filt = epochs[f"{contrast[0]} or {contrast[1]}"]
    else:
        epochs_filt = epochs[contrast]

    # Filtering is costly, so do it last, after the selection of the channels
    # and epochs. We know that often the filter will be longer than the signal,
    # so we ignore the warning here.
    epochs_filt = epochs_filt.filter(fmin, fmax, n_jobs=1, verbose="error")
    y = _prepare_labels(epochs=epochs_filt, contrast=contrast)

    return epochs_filt, y


def get_input_fnames_csp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    contrast: tuple[str],
) -> InFilesT:
    proc = _get_decoding_proc(config=cfg)
    fname_epochs = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        processing=proc,
        suffix="epo",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files = dict()
    in_files["epochs"] = fname_epochs
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


@failsafe_run(get_input_fnames=get_input_fnames_csp)
def one_subject_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str,
    contrast: tuple[str, str],
    in_files: InFilesT,
) -> OutFilesT:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.
    """
    import matplotlib.pyplot as plt

    condition1, condition2 = contrast
    msg = f"Contrasting conditions: {condition1} – {condition2}"
    logger.info(**gen_log_kwargs(msg))

    bids_path = in_files["epochs"].copy().update(processing=None, split=None)
    epochs = mne.read_epochs(in_files.pop("epochs"))
    _restrict_analyze_channels(epochs, cfg)
    pick_idx = mne.pick_types(
        epochs.info, meg=True, eeg=True, ref_meg=False, exclude="bads"
    )
    epochs.pick(pick_idx)

    if cfg.time_frequency_subtract_evoked:
        epochs.subtract_evoked()

    preproc_steps = _decoding_preproc_steps(
        subject=subject,
        session=session,
        epochs=epochs,
    )

    # Classifier
    csp = CSP(
        n_components=4,  # XXX revisit
        reg=0.1,  # XXX revisit
    )
    clf = make_pipeline(
        *preproc_steps,
        csp,
        LogReg(random_state=cfg.random_state),
    )
    cv = StratifiedKFold(
        n_splits=cfg.decoding_n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    # Loop over frequencies (all time points lumped together)
    freq_name_to_bins_map, time_bins = _handle_csp_args(
        cfg.decoding_csp_times,
        cfg.decoding_csp_freqs,
        cfg.decoding_metric,
        epochs_tmin=cfg.epochs_tmin,
        epochs_tmax=cfg.epochs_tmax,
        time_frequency_freq_min=cfg.time_frequency_freq_min,
        time_frequency_freq_max=cfg.time_frequency_freq_max,
    )
    freq_decoding_table_rows = []
    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        for freq_bin in freq_bins:
            f_min, f_max = freq_bin
            row = {
                "subject": [subject],
                "cond_1": [condition1],
                "cond_2": [condition2],
                "f_min": [f_min],
                "f_max": [f_max],
                "freq_range_name": [freq_range_name],
                "mean_crossval_score": [np.nan],
                "scores": [np.ones(5)],
                "metric": [cfg.decoding_metric],
            }
            freq_decoding_table_rows.append(row)

    freq_decoding_table = pd.concat(
        [pd.DataFrame.from_dict(row) for row in freq_decoding_table_rows],
        ignore_index=True,
    )
    del freq_decoding_table_rows

    def _fmt_contrast(
        cond1: str,
        cond2: str,
        fmin: float,
        fmax: float,
        freq_range_name: str,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> str:
        msg = (
            f"Contrast: {cond1} – {cond2}, "
            f"{fmin:4.1f}–{fmax:4.1f} Hz ({freq_range_name})"
        )
        if tmin is not None:
            msg += f" {tmin:+5.3f}–{tmax:+5.3f} sec"
        return msg

    for idx, row in freq_decoding_table.iterrows():
        assert isinstance(row, pd.Series)
        fmin = row["f_min"]
        fmax = row["f_max"]
        cond1 = row["cond_1"]
        cond2 = row["cond_2"]
        freq_range_name = row["freq_range_name"]

        msg = _fmt_contrast(cond1, cond2, fmin, fmax, freq_range_name)
        logger.info(**gen_log_kwargs(msg))

        # XXX We're filtering here again in each iteration. This should be
        # XXX optimized.
        epochs_filt, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg
        )
        # Get the data for all time points
        X = epochs_filt.get_data()

        cv_scores = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            scoring=cfg.decoding_metric,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        freq_decoding_table.loc[idx, "mean_crossval_score"] = cv_scores.mean()
        freq_decoding_table.at[idx, "scores"] = cv_scores
        del fmin, fmax, cond1, cond2, freq_range_name

    # Loop over times x frequencies
    #
    # Note: We don't support varying time ranges for different frequency
    # ranges to avoid leaking of information.
    tf_decoding_table_rows = []

    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        for time_bin in time_bins:
            t_min, t_max = time_bin

            for freq_bin in freq_bins:
                f_min, f_max = freq_bin
                row = {
                    "subject": [subject],
                    "cond_1": [condition1],
                    "cond_2": [condition2],
                    "t_min": [t_min],
                    "t_max": [t_max],
                    "f_min": [f_min],
                    "f_max": [f_max],
                    "freq_range_name": [freq_range_name],
                    "mean_crossval_score": [np.nan],
                    "scores": [np.ones(5, dtype=float)],
                    "metric": [cfg.decoding_metric],
                }
                tf_decoding_table_rows.append(row)

    if len(tf_decoding_table_rows):
        tf_decoding_table = pd.concat(
            [pd.DataFrame.from_dict(row) for row in tf_decoding_table_rows],
            ignore_index=True,
        )
    else:
        tf_decoding_table = pd.DataFrame()
    del tf_decoding_table_rows

    for idx, row in tf_decoding_table.iterrows():
        if len(row) == 0:
            break  # no data
        assert isinstance(row, pd.Series)
        tmin = row["t_min"]
        tmax = row["t_max"]
        fmin = row["f_min"]
        fmax = row["f_max"]
        cond1 = row["cond_1"]
        cond2 = row["cond_2"]
        freq_range_name = row["freq_range_name"]

        epochs_filt, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg
        )
        # Crop data to the time window of interest
        if tmax is not None:  # avoid warnings about outside the interval
            tmax = min(tmax, epochs_filt.times[-1])
        X = epochs_filt.crop(tmin, tmax).get_data()
        del epochs_filt
        cv_scores = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            scoring=cfg.decoding_metric,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        score = cv_scores.mean()
        tf_decoding_table.loc[idx, "mean_crossval_score"] = score
        tf_decoding_table.at[idx, "scores"] = cv_scores
        msg = _fmt_contrast(cond1, cond2, fmin, fmax, freq_range_name, tmin, tmax)
        msg += f": {cfg.decoding_metric}={score:0.3f}"
        logger.info(**gen_log_kwargs(msg))
        del tmin, tmax, fmin, fmax, cond1, cond2, freq_range_name

    # Write each DataFrame to a different Excel worksheet.
    a_vs_b = f"{condition1}+{condition2}".replace(op.sep, "")
    processing = f"{a_vs_b}+CSP+{cfg.decoding_metric}"
    processing = processing.replace("_", "-").replace("-", "")

    fname_results = bids_path.copy().update(
        suffix="decoding", processing=processing, extension=".xlsx"
    )
    with pd.ExcelWriter(fname_results) as w:
        freq_decoding_table.to_excel(w, sheet_name="CSP Frequency", index=False)
        if not tf_decoding_table.empty:
            tf_decoding_table.to_excel(w, sheet_name="CSP Time-Frequency", index=False)
    out_files = {"csp-excel": fname_results}
    del freq_decoding_table

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        msg = "Adding CSP decoding results to the report."
        logger.info(**gen_log_kwargs(message=msg))
        section = "Decoding: CSP"
        all_csp_tf_results = dict()
        for contrast in cfg.decoding_contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f"{cond_1}+{cond_2}".replace(op.sep, "")
            tags = (
                "epochs",
                "contrast",
                "decoding",
                "csp",
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
            )
            processing = f"{a_vs_b}+CSP+{cfg.decoding_metric}"
            processing = processing.replace("_", "-").replace("-", "")
            fname_decoding = bids_path.copy().update(
                processing=processing, suffix="decoding", extension=".xlsx"
            )
            if not fname_decoding.fpath.is_file():
                continue  # not done yet
            csp_freq_results = pd.read_excel(fname_decoding, sheet_name="CSP Frequency")
            csp_freq_results["scores"] = csp_freq_results["scores"].apply(
                lambda x: np.array(x[1:-1].split(), float)
            )
            if not tf_decoding_table.empty:
                csp_tf_results = pd.read_excel(
                    fname_decoding, sheet_name="CSP Time-Frequency"
                )
                csp_tf_results["scores"] = csp_tf_results["scores"].apply(
                    lambda x: np.array(x[1:-1].split(), float)
                )
                all_csp_tf_results[contrast] = csp_tf_results
                del csp_tf_results

            all_decoding_scores = list()
            contrast_names = list()
            for freq_range_name, freq_bins in freq_name_to_bins_map.items():
                results = csp_freq_results.loc[
                    csp_freq_results["freq_range_name"] == freq_range_name
                ]
                results.reset_index(drop=True, inplace=True)
                assert len(results["scores"]) == len(freq_bins)
                for bi, freq_bin in enumerate(freq_bins):
                    all_decoding_scores.append(results["scores"][bi])
                    f_min = float(freq_bin[0])
                    f_max = float(freq_bin[1])
                    contrast_names.append(
                        f"{freq_range_name}\n({f_min:0.1f}-{f_max:0.1f} Hz)"
                    )
            fig, caption, _ = _plot_full_epochs_decoding_scores(
                contrast_names=contrast_names,
                scores=all_decoding_scores,
                metric=cfg.decoding_metric,
            )
            title = f"CSP decoding: {cond_1} vs. {cond_2}"
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                caption=caption,
                tags=tags,
                replace=True,
            )
            # close figure to save memory
            plt.close(fig)
            del fig, caption, title

        # Now, plot decoding scores across time-frequency bins.
        for contrast in cfg.decoding_contrasts:
            if contrast not in all_csp_tf_results:
                continue
            cond_1, cond_2 = contrast
            tags = (
                "epochs",
                "contrast",
                "decoding",
                "csp",
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
            )
            results = all_csp_tf_results[contrast]
            mean_crossval_scores: list[float] = list()
            tmin_list: list[float] = list()
            tmax_list: list[float] = list()
            fmin_list: list[float] = list()
            fmax_list: list[float] = list()
            mean_crossval_scores.extend(
                results["mean_crossval_score"].to_numpy().ravel().tolist()
            )
            tmin_list.extend(results["t_min"].to_numpy().ravel())
            tmax_list.extend(results["t_max"].to_numpy().ravel())
            fmin_list.extend(results["f_min"].to_numpy().ravel())
            fmax_list.extend(results["f_max"].to_numpy().ravel())
            mean_crossval_scores_array = np.array(mean_crossval_scores, float)
            del mean_crossval_scores
            fig, ax = plt.subplots(constrained_layout=True)
            # XXX Add support for more metrics
            assert cfg.decoding_metric == "roc_auc"
            metric = "ROC AUC"
            vmax = (
                max(
                    np.abs(mean_crossval_scores_array.min() - 0.5),
                    np.abs(mean_crossval_scores_array.max() - 0.5),
                )
                + 0.5
            )
            vmin = 0.5 - (vmax - 0.5)
            img = _imshow_tf(
                mean_crossval_scores_array,
                ax,
                tmin=np.array(tmin_list, float),
                tmax=np.array(tmax_list, float),
                fmin=np.array(fmin_list, float),
                fmax=np.array(fmax_list, float),
                vmin=vmin,
                vmax=vmax,
            )
            offset = matplotlib.transforms.offset_copy(
                ax.transData, fig, 6, 0, units="points"
            )
            for freq_range_name, bins in freq_name_to_bins_map.items():
                ax.text(
                    tmin_list[0],
                    0.5 * bins[0][0] + 0.5 * bins[-1][1],
                    freq_range_name,
                    transform=offset,
                    ha="left",
                    va="center",
                    rotation=90,
                )
            ax.set_xlim((np.min(tmin_list), np.max(tmax_list)))
            ax.set_ylim((np.min(fmin_list), np.max(fmax_list)))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            cbar = fig.colorbar(
                ax=ax, shrink=0.75, orientation="vertical", mappable=img
            )
            cbar.set_label(f"Mean decoding score ({metric})")
            title = f"CSP TF decoding: {cond_1} vs. {cond_2}"
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)
            del fig, title

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *, config: SimpleNamespace, subject: str, session: str | None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        # Data parameters
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        # Processing parameters
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked,
        decoding_which_epochs=config.decoding_which_epochs,
        decoding_metric=config.decoding_metric,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        decoding_n_splits=config.decoding_n_splits,
        decoding_contrasts=get_decoding_contrasts(config),
        random_state=config.random_state,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run all subjects decoding in parallel."""
    if not config.contrasts:
        msg = "Skipping, no contrasts specified …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    if not config.decoding_csp:
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ss = _get_ss(config=config)
    ssc = [
        (subject, session, contrast)
        for subject, session in ss
        for contrast in get_decoding_contrasts(config)
    ]
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            one_subject_decoding, exec_params=config.exec_params, n_iter=len(ssc)
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject, session=session),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                contrast=contrast,
            )
            for subject, session, contrast in ssc
        )
        save_logs(logs=logs, config=config)
