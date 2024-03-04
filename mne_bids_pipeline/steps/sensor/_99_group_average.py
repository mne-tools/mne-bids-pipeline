"""Group average at the sensor level.

The M/EEG-channel data are averaged for group averages.
"""

import os
import os.path as op
from functools import partial
from types import SimpleNamespace
from typing import Optional

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.io import loadmat, savemat

from ..._config_utils import (
    _bids_kwargs,
    _pl,
    _restrict_analyze_channels,
    get_decoding_contrasts,
    get_eeg_reference,
    get_sessions,
    get_subjects,
)
from ..._decoding import _handle_csp_args
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import (
    _all_conditions,
    _contrasts_to_names,
    _open_report,
    _plot_decoding_time_generalization,
    _plot_full_epochs_decoding_scores,
    _plot_time_by_time_decoding_scores_gavg,
    _sanitize_cond_tag,
    add_csp_grand_average,
    add_event_counts,
    plot_time_by_time_decoding_t_values,
)
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs
from ...typing import TypedDict


def get_input_fnames_average_evokeds(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[dict],
) -> dict:
    in_files = dict()
    for this_subject in cfg.subjects:
        in_files[f"evoked-{this_subject}"] = BIDSPath(
            subject=this_subject,
            session=session,
            task=cfg.task,
            acquisition=cfg.acq,
            run=None,
            recording=cfg.rec,
            space=cfg.space,
            suffix="ave",
            extension=".fif",
            datatype=cfg.datatype,
            root=cfg.deriv_root,
            check=False,
        )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_average_evokeds,
)
def average_evokeds(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    logger.info(**gen_log_kwargs(message="Creating grand averages"))
    # Container for all conditions:
    conditions = _all_conditions(cfg=cfg)
    evokeds = [list() for _ in range(len(conditions))]

    keys = list(in_files)
    for key in keys:
        if not key.startswith("evoked-"):
            continue
        fname_in = in_files.pop(key)
        these_evokeds = mne.read_evokeds(fname_in)
        for idx, evoked in enumerate(these_evokeds):
            evokeds[idx].append(evoked)  # Insert into the container

    for idx, these_evokeds in enumerate(evokeds):
        evokeds[idx] = mne.grand_average(
            these_evokeds, interpolate_bads=cfg.interpolate_bads_grand_average
        )  # Combine subjects
        # Keep condition in comment
        evokeds[idx].comment = "Grand average: " + these_evokeds[0].comment

    out_files = dict()
    fname_out = out_files["evokeds"] = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        processing="clean",
        recording=cfg.rec,
        space=cfg.space,
        suffix="ave",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )

    if not fname_out.fpath.parent.exists():
        os.makedirs(fname_out.fpath.parent)

    msg = f"Saving grand-averaged evoked sensor data: {fname_out.basename}"
    logger.info(**gen_log_kwargs(message=msg))
    mne.write_evokeds(fname_out, evokeds, overwrite=True)
    if exec_params.interactive:
        for evoked in evokeds:
            evoked.plot()

    # Reporting
    evokeds = [_restrict_analyze_channels(evoked, cfg) for evoked in evokeds]
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        # Add event stats.
        add_event_counts(
            cfg=cfg,
            report=report,
            subject=subject,
            session=session,
        )

        # Evoked responses
        if evokeds:
            n_contrasts = len(cfg.contrasts)
            n_signals = len(evokeds) - n_contrasts
            msg = (
                f"Adding {n_signals} evoked response{_pl(n_signals)} and "
                f"{n_contrasts} contrast{_pl(n_contrasts)} to the report."
            )
        else:
            msg = "No evoked conditions or contrasts found."
        logger.info(**gen_log_kwargs(message=msg))
        for condition, evoked in zip(conditions, evokeds):
            tags = ("evoked", _sanitize_cond_tag(condition))
            if condition in cfg.conditions:
                title = f"Average (sensor): {condition}"
            else:  # It's a contrast of two conditions.
                title = f"Average (sensor) contrast: {condition}"
                tags = tags + ("contrast",)

            report.add_evokeds(
                evokeds=evoked,
                titles=title,
                projs=False,
                tags=tags,
                n_time_points=cfg.report_evoked_n_time_points,
                # captions=evoked.comment,  # TODO upstream
                replace=True,
                n_jobs=1,  # don't auto parallelize
            )

    assert len(in_files) == 0, list(in_files)
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


class ClusterAcrossTime(TypedDict):
    times: np.ndarray
    p_value: float


def _decoding_cluster_permutation_test(
    scores: np.ndarray,
    times: np.ndarray,
    cluster_forming_t_threshold: Optional[float],
    n_permutations: int,
    random_seed: int,
) -> tuple[np.ndarray, list[ClusterAcrossTime], int]:
    """Perform a cluster permutation test on decoding scores.

    The clusters are formed across time points.
    """
    t_vals, all_clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(
        X=scores,
        threshold=cluster_forming_t_threshold,
        n_permutations=n_permutations,
        adjacency=None,  # each time point is "connected" to its neighbors
        out_type="mask",
        tail=1,  # one-sided: significantly above chance level
        seed=random_seed,
        verbose="error",  # ignore No clusters found
    )
    n_permutations = H0.size - 1

    # Convert to a list of Clusters
    clusters = []
    for cluster_idx, cluster_time_slice in enumerate(all_clusters):
        cluster_times = times[cluster_time_slice]
        cluster_p_val = cluster_p_vals[cluster_idx]
        cluster = ClusterAcrossTime(times=cluster_times, p_value=cluster_p_val)
        clusters.append(cluster)

    return t_vals, clusters, n_permutations


def _get_epochs_in_files(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    in_files = dict()
    in_files["epochs"] = BIDSPath(
        subject=cfg.subjects[0],
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix="epo",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


def _decoding_out_fname(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    cond_1: str,
    cond_2: str,
    kind: str,
    extension: str = ".mat",
):
    processing = (
        f"{cond_1}+{cond_2}+{kind}+{cfg.decoding_metric}".replace(op.sep, "")
        .replace("_", "-")
        .replace("-", "")
    )
    return BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        processing=processing,
        suffix="decoding",
        extension=extension,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )


def _get_input_fnames_decoding(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    cond_1: str,
    cond_2: str,
    kind: str,
    extension: str = ".mat",
) -> dict:
    in_files = _get_epochs_in_files(cfg=cfg, subject=subject, session=session)
    for this_subject in cfg.subjects:
        in_files[f"scores-{this_subject}"] = _decoding_out_fname(
            cfg=cfg,
            subject=this_subject,
            session=session,
            cond_1=cond_1,
            cond_2=cond_2,
            kind=kind,
            extension=extension,
        )
    return in_files


@failsafe_run(
    get_input_fnames=partial(
        _get_input_fnames_decoding,
        kind="TimeByTime",
    ),
)
def average_time_by_time_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    cond_1: str,
    cond_2: str,
    in_files: dict,
) -> dict:
    logger.info(**gen_log_kwargs(message="Averaging time-by-time decoding results"))
    # Get the time points from the very first subject. They are identical
    # across all subjects and conditions, so this should suffice.
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=False)
    dtg_decim = cfg.decoding_time_generalization_decim
    if cfg.decoding_time_generalization and dtg_decim > 1:
        epochs.decimate(dtg_decim, verbose="error")
    times = epochs.times
    del epochs

    if cfg.decoding_time_generalization:
        time_points_shape = (len(times), len(times))
    else:
        time_points_shape = (len(times),)

    n_subjects = len(cfg.subjects)
    contrast_score_stats = {
        "cond_1": cond_1,
        "cond_2": cond_2,
        "times": times,
        "N": n_subjects,
        "decim": dtg_decim,
        "mean": np.empty(time_points_shape),
        "mean_min": np.empty(time_points_shape),
        "mean_max": np.empty(time_points_shape),
        "mean_se": np.empty(time_points_shape),
        "mean_ci_lower": np.empty(time_points_shape),
        "mean_ci_upper": np.empty(time_points_shape),
        "cluster_all_times": np.array([]),
        "cluster_all_t_values": np.array([]),
        "cluster_t_threshold": np.nan,
        "cluster_n_permutations": np.nan,
        "clusters": list(),
    }

    # Extract mean CV scores from all subjects.
    mean_scores = np.empty((n_subjects, *time_points_shape))

    # Remaining in_files are all decoding data
    assert len(in_files) == n_subjects, list(in_files.keys())
    for sub_idx, key in enumerate(list(in_files)):
        decoding_data = loadmat(in_files.pop(key))
        mean_scores[sub_idx, :] = decoding_data["scores"].mean(axis=0)

    # Cluster permutation test.
    # We can only permute for two or more subjects
    #
    # If we've performed time generalization, we will only use the diagonal
    # CV scores here (classifiers trained and tested at the same time
    # points).

    if n_subjects > 1:
        # Constrain cluster permutation test to time points of the
        # time-locked event or later.
        # We subtract the chance level from the scores as we'll be
        # performing a 1-sample test (i.e., test against 0)!
        idx = np.where(times >= 0)[0]

        if cfg.decoding_time_generalization:
            cluster_permutation_scores = mean_scores[:, idx, idx] - 0.5
        else:
            cluster_permutation_scores = mean_scores[:, idx] - 0.5

        cluster_permutation_times = times[idx]
        if cfg.cluster_forming_t_threshold is None:
            import scipy.stats

            cluster_forming_t_threshold = scipy.stats.t.ppf(
                1 - 0.05, len(cluster_permutation_scores) - 1
            )
        else:
            cluster_forming_t_threshold = cfg.cluster_forming_t_threshold

        t_vals, clusters, n_perm = _decoding_cluster_permutation_test(
            scores=cluster_permutation_scores,
            times=cluster_permutation_times,
            cluster_forming_t_threshold=cluster_forming_t_threshold,
            n_permutations=cfg.cluster_n_permutations,
            random_seed=cfg.random_state,
        )

        contrast_score_stats.update(
            {
                "cluster_all_times": cluster_permutation_times,
                "cluster_all_t_values": t_vals,
                "cluster_t_threshold": cluster_forming_t_threshold,
                "clusters": clusters,
                "cluster_n_permutations": n_perm,
            }
        )

        del cluster_permutation_scores, cluster_permutation_times, n_perm

    # Now we can calculate some descriptive statistics on the mean scores.
    # We use the [:] here as a safeguard to ensure we don't mess up the
    # dimensions.
    #
    # For time generalization, all values (each time point vs each other)
    # are considered.
    contrast_score_stats["mean"][:] = mean_scores.mean(axis=0)
    contrast_score_stats["mean_min"][:] = mean_scores.min(axis=0)
    contrast_score_stats["mean_max"][:] = mean_scores.max(axis=0)

    # Finally, for each time point, bootstrap the mean, and calculate the
    # SD of the bootstrapped distribution: this is the standard error of
    # the mean. We also derive 95% confidence intervals.
    rng = np.random.default_rng(seed=cfg.random_state)
    for time_idx in range(len(times)):
        if cfg.decoding_time_generalization:
            data = mean_scores[:, time_idx, time_idx]
        else:
            data = mean_scores[:, time_idx]
        scores_resampled = rng.choice(data, size=(cfg.n_boot, n_subjects), replace=True)
        bootstrapped_means = scores_resampled.mean(axis=1)

        # SD of the bootstrapped distribution == SE of the metric.
        se = bootstrapped_means.std(ddof=1)
        ci_lower = np.quantile(bootstrapped_means, q=0.025)
        ci_upper = np.quantile(bootstrapped_means, q=0.975)

        contrast_score_stats["mean_se"][time_idx] = se
        contrast_score_stats["mean_ci_lower"][time_idx] = ci_lower
        contrast_score_stats["mean_ci_upper"][time_idx] = ci_upper

        del bootstrapped_means, se, ci_lower, ci_upper

    out_files = dict()
    out_files["mat"] = _decoding_out_fname(
        cfg=cfg,
        subject=subject,
        session=session,
        cond_1=cond_1,
        cond_2=cond_2,
        kind="TimeByTime",
    )
    savemat(out_files["mat"], contrast_score_stats)

    section = "Decoding: time-by-time"
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        logger.info(**gen_log_kwargs(message="Adding time-by-time decoding results"))
        import matplotlib.pyplot as plt

        tags = (
            "epochs",
            "contrast",
            "decoding",
            f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
        )
        decoding_data = loadmat(out_files["mat"])

        # Plot scores
        fig = _plot_time_by_time_decoding_scores_gavg(
            cfg=cfg,
            decoding_data=decoding_data,
        )
        caption = (
            f'Based on N={decoding_data["N"].squeeze()} '
            f"subjects. Standard error and confidence interval "
            f"of the mean were bootstrapped with {cfg.n_boot} "
            f"resamples. CI must not be used for statistical inference here, "
            f"as it is not corrected for multiple testing."
        )
        if len(get_subjects(cfg)) > 1:
            caption += (
                f" Time periods with decoding performance significantly above "
                f"chance, if any, were derived with a one-tailed "
                f"cluster-based permutation test "
                f'({decoding_data["cluster_n_permutations"].squeeze()} '
                f"permutations) and are highlighted in yellow."
            )
            title = f"Decoding over time: {cond_1} vs. {cond_2}"
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)

        # Plot t-values used to form clusters
        if len(get_subjects(cfg)) > 1:
            fig = plot_time_by_time_decoding_t_values(decoding_data=decoding_data)
            t_threshold = np.round(decoding_data["cluster_t_threshold"], 3).item()
            caption = (
                f"Observed t-values. Time points with "
                f"t-values > {t_threshold} were used to form clusters."
            )
            report.add_figure(
                fig=fig,
                title=f"t-values across time: {cond_1} vs. {cond_2}",
                caption=caption,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)

        if cfg.decoding_time_generalization:
            fig = _plot_decoding_time_generalization(
                decoding_data=decoding_data,
                metric=cfg.decoding_metric,
                kind="grand-average",
            )
            caption = (
                f"Time generalization (generalization across time, GAT): "
                f"each classifier is trained on each time point, and tested "
                f"on all other time points. The results were averaged across "
                f'N={decoding_data["N"].item()} subjects.'
            )
            title = f"Time generalization: {cond_1} vs. {cond_2}"
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)

    return _prep_out_files(out_files=out_files, exec_params=exec_params)


@failsafe_run(
    get_input_fnames=partial(
        _get_input_fnames_decoding,
        kind="FullEpochs",
    ),
)
def average_full_epochs_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    cond_1: str,
    cond_2: str,
    in_files: dict,
) -> dict:
    n_subjects = len(cfg.subjects)
    in_files.pop("epochs")  # not used but okay to include

    contrast_score_stats = {
        "cond_1": cond_1,
        "cond_2": cond_2,
        "N": n_subjects,
        "subjects": cfg.subjects,
        "scores": np.nan,
        "mean": np.nan,
        "mean_min": np.nan,
        "mean_max": np.nan,
        "mean_se": np.nan,
        "mean_ci_lower": np.nan,
        "mean_ci_upper": np.nan,
    }

    # Extract mean CV scores from all subjects.
    mean_scores = np.empty(n_subjects)
    for sub_idx, key in enumerate(list(in_files)):
        decoding_data = loadmat(in_files.pop(key))
        mean_scores[sub_idx] = decoding_data["scores"].mean()

    # Now we can calculate some descriptive statistics on the mean scores.
    # We use the [:] here as a safeguard to ensure we don't mess up the
    # dimensions.
    contrast_score_stats["scores"] = mean_scores
    contrast_score_stats["mean"] = mean_scores.mean()
    contrast_score_stats["mean_min"] = mean_scores.min()
    contrast_score_stats["mean_max"] = mean_scores.max()

    # Finally, bootstrap the mean, and calculate the
    # SD of the bootstrapped distribution: this is the standard error of
    # the mean. We also derive 95% confidence intervals.
    rng = np.random.default_rng(seed=cfg.random_state)
    scores_resampled = rng.choice(
        mean_scores, size=(cfg.n_boot, n_subjects), replace=True
    )
    bootstrapped_means = scores_resampled.mean(axis=1)

    # SD of the bootstrapped distribution == SE of the metric.
    se = bootstrapped_means.std(ddof=1)
    ci_lower = np.quantile(bootstrapped_means, q=0.025)
    ci_upper = np.quantile(bootstrapped_means, q=0.975)

    contrast_score_stats["mean_se"] = se
    contrast_score_stats["mean_ci_lower"] = ci_lower
    contrast_score_stats["mean_ci_upper"] = ci_upper

    del bootstrapped_means, se, ci_lower, ci_upper

    out_files = dict()
    fname_out = out_files["mat"] = _decoding_out_fname(
        cfg=cfg,
        subject=subject,
        session=session,
        cond_1=cond_1,
        cond_2=cond_2,
        kind="FullEpochs",
    )
    if not fname_out.fpath.parent.exists():
        os.makedirs(fname_out.fpath.parent)
    savemat(fname_out, contrast_score_stats)
    return _prep_out_files(out_files=out_files, exec_params=exec_params)


def get_input_files_average_full_epochs_report(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    decoding_contrasts: list[list[str]],
) -> dict:
    in_files = dict()
    for contrast in decoding_contrasts:
        in_files[f"decoding-full-epochs-{contrast}"] = _decoding_out_fname(
            cfg=cfg,
            subject=subject,
            session=session,
            cond_1=contrast[0],
            cond_2=contrast[1],
            kind="FullEpochs",
        )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_files_average_full_epochs_report,
)
def average_full_epochs_report(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    decoding_contrasts: list[list[str]],
    in_files: dict,
) -> dict:
    """Add decoding results to the grand average report."""
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        import matplotlib.pyplot as plt  # nested import to help joblib

        logger.info(
            **gen_log_kwargs(message="Adding full-epochs decoding results to report")
        )

        # Full-epochs decoding
        all_decoding_scores = []
        for key in list(in_files):
            if not key.startswith("decoding-full-epochs-"):
                continue
            decoding_data = loadmat(in_files.pop(key))
            all_decoding_scores.append(np.atleast_1d(decoding_data["scores"].squeeze()))
            del decoding_data

        fig, caption = _plot_full_epochs_decoding_scores(
            contrast_names=_contrasts_to_names(decoding_contrasts),
            scores=all_decoding_scores,
            metric=cfg.decoding_metric,
            kind="grand-average",
        )
        report.add_figure(
            fig=fig,
            title="Full-epochs decoding",
            section="Decoding: full-epochs",
            caption=caption,
            tags=(
                "epochs",
                "contrast",
                "decoding",
                *[
                    f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}"
                    for cond_1, cond_2 in cfg.decoding_contrasts
                ],
            ),
            replace=True,
        )
        # close figure to save memory
        plt.close(fig)
    return _prep_out_files(exec_params=exec_params, out_files=dict())


@failsafe_run(
    get_input_fnames=partial(
        _get_input_fnames_decoding,
        kind="CSP",
        extension=".xlsx",
    ),
)
def average_csp_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    cond_1: str,
    cond_2: str,
    in_files: dict,
):
    msg = f"Summarizing CSP results: {cond_1} - {cond_2}."
    logger.info(**gen_log_kwargs(message=msg))
    in_files.pop("epochs")

    all_decoding_data_freq = []
    all_decoding_data_time_freq = []
    for key in list(in_files):
        fname_xlsx = in_files.pop(key)
        decoding_data_freq = pd.read_excel(
            fname_xlsx,
            sheet_name="CSP Frequency",
            dtype={"subject": str},  # don't drop trailing zeros
        )
        decoding_data_time_freq = pd.read_excel(
            fname_xlsx,
            sheet_name="CSP Time-Frequency",
            dtype={"subject": str},  # don't drop trailing zeros
        )
        all_decoding_data_freq.append(decoding_data_freq)
        all_decoding_data_time_freq.append(decoding_data_time_freq)
        del fname_xlsx

    # Now calculate descriptes and bootstrap CIs.
    grand_average_freq = _average_csp_time_freq(
        cfg=cfg,
        subject=subject,
        session=session,
        data=all_decoding_data_freq,
    )
    grand_average_time_freq = _average_csp_time_freq(
        cfg=cfg,
        subject=subject,
        session=session,
        data=all_decoding_data_time_freq,
    )

    out_files = dict()
    out_files["freq"] = _decoding_out_fname(
        cfg=cfg,
        subject=subject,
        session=session,
        cond_1=cond_1,
        cond_2=cond_2,
        kind="CSP",
        extension=".xlsx",
    )
    with pd.ExcelWriter(out_files["freq"]) as w:
        grand_average_freq.to_excel(w, sheet_name="CSP Frequency", index=False)
        grand_average_time_freq.to_excel(
            w, sheet_name="CSP Time-Frequency", index=False
        )

    # Perform a cluster-based permutation test.
    subjects = cfg.subjects
    time_bins = np.array(cfg.decoding_csp_times)
    if time_bins.ndim == 1:
        time_bins = np.array(list(zip(time_bins[:-1], time_bins[1:])))
    time_bins = pd.DataFrame(time_bins, columns=["t_min", "t_max"])
    freq_name_to_bins_map = _handle_csp_args(
        cfg.decoding_csp_times,
        cfg.decoding_csp_freqs,
        cfg.decoding_metric,
        epochs_tmin=cfg.epochs_tmin,
        epochs_tmax=cfg.epochs_tmax,
        time_frequency_freq_min=cfg.time_frequency_freq_min,
        time_frequency_freq_max=cfg.time_frequency_freq_max,
    )
    data_for_clustering = {}
    for freq_range_name in freq_name_to_bins_map:
        a = np.empty(
            shape=(
                len(subjects),
                len(time_bins),
                len(freq_name_to_bins_map[freq_range_name]),
            )
        )
        a.fill(np.nan)
        data_for_clustering[freq_range_name] = a

    g = pd.concat(all_decoding_data_time_freq).groupby(
        ["subject", "freq_range_name", "t_min", "t_max"]
    )

    for (subject_, freq_range_name, t_min, t_max), df in g:
        scores = df["mean_crossval_score"]
        sub_idx = subjects.index(subject_)
        time_bin_idx = time_bins.loc[
            (np.isclose(time_bins["t_min"], t_min))
            & (np.isclose(time_bins["t_max"], t_max)),
            :,
        ].index
        assert len(time_bin_idx) == 1
        time_bin_idx = time_bin_idx[0]
        data_for_clustering[freq_range_name][sub_idx][time_bin_idx] = scores

    if cfg.cluster_forming_t_threshold is None:
        import scipy.stats

        cluster_forming_t_threshold = scipy.stats.t.ppf(
            1 - 0.05,
            len(cfg.subjects) - 1,  # one-sided test
        )
    else:
        cluster_forming_t_threshold = cfg.cluster_forming_t_threshold

    cluster_permutation_results = {}
    for freq_range_name, X in data_for_clustering.items():
        if len(X) < 2:
            t_vals = np.full(X.shape[1:], np.nan)
            H0 = all_clusters = cluster_p_vals = np.array([])
        else:
            (
                t_vals,
                all_clusters,
                cluster_p_vals,
                H0,
            ) = mne.stats.permutation_cluster_1samp_test(  # noqa: E501
                X=X - 0.5,  # One-sample test against zero.
                threshold=cluster_forming_t_threshold,
                n_permutations=cfg.cluster_n_permutations,
                adjacency=None,  # each time & freq bin connected to its neighbors
                out_type="mask",
                tail=1,  # one-sided: significantly above chance level
                seed=cfg.random_state,
            )
        n_permutations = H0.size - 1
        all_clusters = np.array(all_clusters)  # preserve "empty" 0th dimension
        cluster_permutation_results[freq_range_name] = {
            "mean_crossval_scores": X.mean(axis=0),
            "t_vals": t_vals,
            "clusters": all_clusters,
            "cluster_p_vals": cluster_p_vals,
            "cluster_t_threshold": cluster_forming_t_threshold,
            "n_permutations": n_permutations,
            "time_bin_edges": cfg.decoding_csp_times,
            "freq_bin_edges": cfg.decoding_csp_freqs[freq_range_name],
        }

    out_files["cluster"] = out_files["freq"].copy().update(extension=".mat")
    savemat(file_name=out_files["cluster"], mdict=cluster_permutation_results)

    assert subject == "average"
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        add_csp_grand_average(
            cfg=cfg,
            subject=subject,
            session=session,
            report=report,
            cond_1=cond_1,
            cond_2=cond_2,
            fname_csp_freq_results=out_files["freq"],
            fname_csp_cluster_results=out_files["cluster"],
        )
    return _prep_out_files(out_files=out_files, exec_params=exec_params)


def _average_csp_time_freq(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    data: pd.DataFrame,
) -> pd.DataFrame:
    # Prepare a dataframe for storing the results.
    grand_average = data[0].copy()
    del grand_average["mean_crossval_score"]

    grand_average["subject"] = subject
    grand_average["mean"] = np.nan
    grand_average["mean_se"] = np.nan
    grand_average["mean_ci_lower"] = np.nan
    grand_average["mean_ci_upper"] = np.nan

    # Now generate descriptive and bootstrapped statistics.
    n_subjects = len(cfg.subjects)
    rng = np.random.default_rng(seed=cfg.random_state)
    for row_idx, row in grand_average.iterrows():
        all_scores = np.array([df.loc[row_idx, "mean_crossval_score"] for df in data])

        grand_average.loc[row_idx, "mean"] = all_scores.mean()

        # Abort here if we only have a single subject – no need to bootstrap
        # CIs etc.
        if len(cfg.subjects) == 1:
            continue

        # Bootstrap the mean, and calculate the
        # SD of the bootstrapped distribution: this is the standard error of
        # the mean. We also derive 95% confidence intervals.
        scores_resampled = rng.choice(
            all_scores, size=(cfg.n_boot, n_subjects), replace=True
        )
        bootstrapped_means = scores_resampled.mean(axis=1)

        # SD of the bootstrapped distribution == SE of the metric.
        with np.errstate(over="raise"):
            se = bootstrapped_means.std(ddof=1)
        ci_lower = np.quantile(bootstrapped_means, q=0.025)
        ci_upper = np.quantile(bootstrapped_means, q=0.975)

        grand_average.loc[row_idx, "mean_se"] = se
        grand_average.loc[row_idx, "mean_ci_lower"] = ci_lower
        grand_average.loc[row_idx, "mean_ci_upper"] = ci_upper

        del (
            bootstrapped_means,
            se,
            ci_lower,
            ci_upper,
            scores_resampled,
            all_scores,
            row_idx,
            row,
        )

    return grand_average


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        subjects=get_subjects(config),
        task_is_rest=config.task_is_rest,
        conditions=config.conditions,
        contrasts=config.contrasts,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_n_splits=config.decoding_n_splits,
        decoding_time_generalization=config.decoding_time_generalization,
        decoding_time_generalization_decim=config.decoding_time_generalization_decim,
        decoding_csp=config.decoding_csp,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        decoding_contrasts=get_decoding_contrasts(config),
        random_state=config.random_state,
        n_boot=config.n_boot,
        cluster_forming_t_threshold=config.cluster_forming_t_threshold,
        cluster_n_permutations=config.cluster_n_permutations,
        analyze_channels=config.analyze_channels,
        interpolate_bads_grand_average=config.interpolate_bads_grand_average,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        sessions=get_sessions(config),
        exclude_subjects=config.exclude_subjects,
        report_evoked_n_time_points=config.report_evoked_n_time_points,
        cluster_permutation_p_threshold=config.cluster_permutation_p_threshold,
        # TODO: needed because get_datatype gets called again...
        data_type=config.data_type,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    if config.task_is_rest:
        msg = '    … skipping: for "rest" task.'
        logger.info(**gen_log_kwargs(message=msg))
        return
    cfg = get_config(
        config=config,
    )
    exec_params = config.exec_params
    subject = "average"
    sessions = get_sessions(config=config)
    if cfg.decode or cfg.decoding_csp:
        decoding_contrasts = get_decoding_contrasts(config=cfg)
    else:
        decoding_contrasts = []
    logs = list()
    with get_parallel_backend(exec_params):
        # 1. Evoked data
        logs += [
            average_evokeds(
                cfg=cfg,
                exec_params=exec_params,
                subject=subject,
                session=session,
            )
            for session in sessions
        ]

        # 2. Time decoding
        if cfg.decode and decoding_contrasts:
            # Full epochs (single report function plots across all contrasts
            # so it's a separate cached step)
            logs += [
                average_full_epochs_decoding(
                    cfg=cfg,
                    subject=subject,
                    session=session,
                    cond_1=contrast[0],
                    cond_2=contrast[1],
                    exec_params=exec_params,
                )
                for session in sessions
                for contrast in decoding_contrasts
            ]
            logs += [
                average_full_epochs_report(
                    cfg=cfg,
                    exec_params=exec_params,
                    subject=subject,
                    session=session,
                    decoding_contrasts=decoding_contrasts,
                )
                for session in sessions
            ]
            # Time-by-time
            parallel, run_func = parallel_func(
                average_time_by_time_decoding, exec_params=exec_params
            )
            logs += parallel(
                run_func(
                    cfg=cfg,
                    exec_params=exec_params,
                    subject=subject,
                    session=session,
                    cond_1=contrast[0],
                    cond_2=contrast[1],
                )
                for session in sessions
                for contrast in decoding_contrasts
            )

        # 3. CSP
        if cfg.decoding_csp and decoding_contrasts:
            parallel, run_func = parallel_func(
                average_csp_decoding, exec_params=exec_params
            )
            logs += parallel(
                run_func(
                    cfg=cfg,
                    exec_params=exec_params,
                    subject=subject,
                    session=session,
                    cond_1=contrast[0],
                    cond_2=contrast[1],
                )
                for contrast in get_decoding_contrasts(config=cfg)
                for session in sessions
            )

    save_logs(config=config, logs=logs)
