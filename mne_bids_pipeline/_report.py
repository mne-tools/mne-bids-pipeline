import contextlib
import traceback
from collections.abc import Generator
from functools import lru_cache
from io import StringIO
from textwrap import indent
from types import SimpleNamespace
from typing import Any, Literal

import matplotlib.axes
import matplotlib.figure
import matplotlib.image
import matplotlib.transforms
import mne
import numpy as np
import pandas as pd
from filelock import FileLock
from mne.io import BaseRaw
from mne.report.report import _df_bootstrap_table
from mne.utils import _pl
from mne_bids import BIDSPath
from mne_bids.stats import count_events
from scipy.io import loadmat

from ._config_utils import _get_task_contrasts
from ._decoding import _handle_csp_args
from ._logging import _linkfile, gen_log_kwargs, logger
from .typing import FloatArrayT


def _report_path(
    *, cfg: SimpleNamespace, subject: str, session: str | None = None
) -> BIDSPath:
    return BIDSPath(
        subject=subject,
        session=session,
        # Report is across all runs and tasks, but for logging purposes it's helpful
        # to pass the run and task for gen_log_kwargs
        run=None,
        task=None,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        extension=".h5",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        suffix="report",
        check=False,
    )


@contextlib.contextmanager
def _open_report(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None = None,
    task: str | None = None,
    fname_report: BIDSPath | None = None,
    name: str = "report",
) -> Generator[mne.Report, None, None]:
    if fname_report is None:
        fname_report = _report_path(cfg=cfg, subject=subject, session=session)
    fname_report = fname_report.fpath
    assert fname_report.suffix == ".h5", fname_report.suffix
    # prevent parallel file access
    with FileLock(f"{fname_report}.lock"), _agg_backend():
        if not fname_report.is_file():
            msg = f"Initializing {name} HDF5 file"
            logger.info(**gen_log_kwargs(message=msg))
            report = _gen_empty_report(
                cfg=cfg,
                subject=subject,
                session=session,
            )
            report.save(fname_report)
        try:
            report = mne.open_report(fname_report)
        except Exception as exc:
            raise exc.__class__(
                f"Could not open {name} HDF5 file:\n{fname_report}, "
                "Perhaps you need to delete it? Got error:\n\n"
                f"{indent(traceback.format_exc(), '    ')}"
            ) from None
        try:
            yield report
        finally:
            try:
                _finalize(
                    cfg=cfg,
                    report=report,
                    exec_params=exec_params,
                    subject=subject,
                    session=session,
                    run=run,
                    task=task,
                )
            except Exception as exc:
                logger.warning(f"Failed: {exc}")
            fname_report_html = fname_report.with_suffix(".html")
            msg = f"Saving {name}: {_linkfile(fname_report_html)}"
            logger.info(**gen_log_kwargs(message=msg), sanitize=False)
            report.save(fname_report, overwrite=True)
            report.save(fname_report_html, overwrite=True, open_browser=False)


# def plot_full_epochs_decoding_scores(
#     contrast: str,
#     cross_val_scores: np.ndarray,
#     metric: str
# ):
#     """Plot cross-validation results from full-epochs decoding.
#     """
#     import matplotlib.pyplot as plt  # nested import to help joblib
#     import seaborn as sns

#     cross_val_scores = cross_val_scores.squeeze()  # Make it a 1D array
#     data = pd.DataFrame({
#         'contrast': [contrast] * len(cross_val_scores),
#         'scores': cross_val_scores,
#         'metric': [metric] * len(cross_val_scores)}
#     )
#     fig, ax = plt.subplots(constrained_layout=True)

#     sns.swarmplot(x='contrast', y='scores', data=data, color='0.25',
#                   label='cross-val. scores')
#     ax.set_xticklabels([])

#     ax.plot(cross_val_scores.mean(), '+', color='red', ms=15,
#             label='mean score', zorder=99)
#     ax.axhline(0.5, ls='--', lw=0.5, color='black', label='chance')

#     ax.set_xlabel(f'{contrast[0]} vs. {contrast[1]}')
#     if metric == 'roc_auc':
#         metric = 'ROC AUC'
#     ax.set_ylabel(f'Score ({metric})')
#     ax.legend(loc='center right')

#     return fig


def _plot_full_epochs_decoding_scores(
    contrast_names: list[str],
    scores: list[FloatArrayT],
    metric: str,
    kind: Literal["single-subject", "grand-average"] = "single-subject",
) -> tuple[matplotlib.figure.Figure, str, pd.DataFrame]:
    """Plot cross-validation results from full-epochs decoding."""
    import matplotlib.pyplot as plt  # nested import to help joblib
    import seaborn as sns

    if metric == "roc_auc":
        metric = "ROC AUC"
    score_label = f"Score ({metric})"

    data = pd.DataFrame(
        {
            "Contrast": np.array(
                [[c] * len(scores[0]) for c in contrast_names]
            ).flatten(),
            score_label: np.hstack(scores),
        }
    )

    if kind == "grand-average":
        # First create a grid of boxplots …
        g = sns.catplot(
            data=data,
            y=score_label,
            kind="box",
            col="Contrast",
            col_wrap=3,
            aspect=0.33,
        )
        # … and now add swarmplots on top to visualize every single data point.
        g.map_dataframe(sns.swarmplot, y=score_label, color="black")
        caption = (
            f"Based on N={len(scores[0])} "
            f"subjects. Each dot represents the mean cross-validation score "
            f"for a single subject. The dashed line is expected chance "
            f"performance."
        )
    else:
        # First create a grid of swarmplots to visualize every single
        # cross-validation score.
        g = sns.catplot(
            data=data,
            y=score_label,
            kind="swarm",
            col="Contrast",
            col_wrap=3,
            aspect=0.33,
            color="black",
        )

        # And now add the mean CV score on top.
        def _plot_mean_cv_score(x: FloatArrayT, **kwargs: dict[str, Any]) -> None:
            plt.plot(x.mean(), **kwargs)

        g.map(
            _plot_mean_cv_score,
            score_label,
            marker="+",
            color="red",
            ms=15,
            label="mean score",
            zorder=99,
        )
        caption = (
            "Each black dot represents the single cross-validation score. "
            f"The red cross is the mean of all {len(scores[0])} "
            "cross-validation scores. "
            "The dashed line is expected chance performance."
        )
        plt.xlim([-0.1, 0.1])

    g.map(plt.axhline, y=0.5, ls="--", lw=0.5, color="black", zorder=99)
    g.set_titles("{col_name}")  # use this argument literally!
    g.set_xlabels("")

    fig = g.fig
    return fig, caption, data


def _plot_time_by_time_decoding_scores(
    *,
    times: FloatArrayT,
    cross_val_scores: FloatArrayT,
    metric: str,
    time_generalization: bool,
    decim: int,
) -> matplotlib.figure.Figure:
    """Plot cross-validation results from time-by-time decoding."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    mean_scores = cross_val_scores.mean(axis=0)
    max_scores = cross_val_scores.max(axis=0)
    min_scores = cross_val_scores.min(axis=0)

    if time_generalization:
        # Only use the diagonal values (classifiers trained and tested on the
        # same time points).
        mean_scores = np.diag(mean_scores)
        max_scores = np.diag(max_scores)
        min_scores = np.diag(min_scores)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.axhline(0.5, ls="--", lw=0.5, color="black", label="chance")
    if times.min() < 0 < times.max():
        ax.axvline(0, ls="-", lw=0.5, color="black")
    ax.fill_between(
        x=times,
        y1=min_scores,
        y2=max_scores,
        color="lightgray",
        alpha=0.5,
        label="range [min, max]",
    )
    ax.plot(times, mean_scores, ls="-", lw=2, color="black", label="mean")

    _label_time_by_time(ax, xlabel="Time (s)", decim=decim)
    if metric == "roc_auc":
        metric = "ROC AUC"
    ax.set_ylabel(f"Score ({metric})")
    ax.set_ylim((-0.025, 1.025))
    ax.legend(loc="lower right")

    return fig


def _label_time_by_time(
    ax: matplotlib.axes.Axes,
    *,
    decim: int,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    extra = ""
    if decim > 1:
        extra = f" (decim={decim})"
    if xlabel is not None:
        ax.set_xlabel(f"{xlabel}{extra}")
    if ylabel is not None:
        ax.set_ylabel(f"{ylabel}{extra}")


def _plot_time_by_time_decoding_scores_gavg(
    *, cfg: SimpleNamespace, decoding_data: dict[str, Any]
) -> matplotlib.figure.Figure:
    """Plot the grand-averaged decoding scores."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    times = decoding_data["times"].squeeze()
    mean_scores = decoding_data["mean"].squeeze()
    se_lower = mean_scores - decoding_data["mean_se"].squeeze()
    se_upper = mean_scores + decoding_data["mean_se"].squeeze()
    ci_lower = decoding_data["mean_ci_lower"].squeeze()
    ci_upper = decoding_data["mean_ci_upper"].squeeze()
    decim = decoding_data["decim"].item()

    if cfg.decoding_time_generalization:
        # Only use the diagonal values (classifiers trained and tested on the
        # same time points).
        mean_scores = np.diag(mean_scores)
        se_lower = np.diag(se_lower)
        se_upper = np.diag(se_upper)
        ci_lower = np.diag(ci_lower)
        ci_upper = np.diag(ci_upper)

    metric = cfg.decoding_metric
    clusters = np.atleast_1d(decoding_data["clusters"].squeeze())

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ylim((-0.025, 1.025))

    # Start with plotting the significant time periods according to the
    # cluster-based permutation test
    n_significant_clusters_plotted = 0
    for cluster in clusters:
        cluster_times = np.atleast_1d(cluster["times"][0][0].squeeze())
        cluster_p = cluster["p_value"][0][0].item()
        if cluster_p >= cfg.cluster_permutation_p_threshold:
            continue

        # Only add the label once
        if n_significant_clusters_plotted == 0:
            label = f"$p$ < {cfg.cluster_permutation_p_threshold} (cluster pemutation)"
        else:
            label = None

        ax.fill_betweenx(
            y=ax.get_ylim(),
            x1=cluster_times[0],
            x2=cluster_times[-1],
            facecolor="orange",
            alpha=0.15,
            label=label,
        )
        n_significant_clusters_plotted += 1

    ax.axhline(0.5, ls="--", lw=0.5, color="black", label="chance")
    if times.min() < 0 < times.max():
        ax.axvline(0, ls="-", lw=0.5, color="black")
    ax.fill_between(
        x=times,
        y1=ci_lower,
        y2=ci_upper,
        color="lightgray",
        alpha=0.5,
        label="95% confidence interval",
    )

    ax.plot(times, mean_scores, ls="-", lw=2, color="black", label="mean")
    ax.plot(
        times, se_lower, ls="-.", lw=0.5, color="gray", label="mean ± standard error"
    )
    ax.plot(times, se_upper, ls="-.", lw=0.5, color="gray")
    ax.text(
        0.05,
        0.05,
        s=f"$N$={decoding_data['N'].squeeze()}",
        fontsize="x-large",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    _label_time_by_time(ax, xlabel="Time (s)", decim=decim)
    if metric == "roc_auc":
        metric = "ROC AUC"
    ax.set_ylabel(f"Score ({metric})")
    ax.legend(loc="lower right")

    return fig


def plot_time_by_time_decoding_t_values(
    decoding_data: dict[str, Any],
) -> matplotlib.figure.Figure:
    """Plot the t-values used to form clusters for the permutation test."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    all_times = decoding_data["cluster_all_times"].squeeze()
    all_t_values = decoding_data["cluster_all_t_values"].squeeze()
    t_threshold = decoding_data["cluster_t_threshold"].item()
    decim = decoding_data["decim"].item()

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(all_times, all_t_values, ls="-", color="black", label="observed $t$-values")
    ax.axhline(t_threshold, ls="--", color="red", label="threshold")

    ax.text(
        0.05,
        0.05,
        s=f"$N$={decoding_data['N'].squeeze()}",
        fontsize="x-large",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )

    _label_time_by_time(ax, xlabel="Time (s)", decim=decim)
    ax.set_ylabel("$t$-value")
    ax.legend(loc="lower right")

    if all_t_values.min() < 0 and all_t_values.max() > 0:
        # center the y axis around 0
        y_max = np.abs(ax.get_ylim()).max()
        ax.set_ylim(ymin=-y_max, ymax=y_max)
    elif all_t_values.min() > 0 and all_t_values.max() > 0:
        # start y axis at zero
        ax.set_ylim(ymin=0, ymax=all_t_values.max())
    elif all_t_values.min() < 0 and all_t_values.max() < 0:
        # start y axis at zero
        ax.set_ylim(ymin=all_t_values.min(), ymax=0)

    return fig


def _plot_decoding_time_generalization(
    decoding_data: dict[str, Any],
    metric: str,
    kind: Literal["single-subject", "grand-average"],
) -> matplotlib.figure.Figure:
    """Plot time generalization matrix."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    times = decoding_data["times"].squeeze()
    decim = decoding_data["decim"].item()
    if kind == "single-subject":
        # take the mean across CV scores
        mean_scores = decoding_data["scores"].mean(axis=0)
    else:
        mean_scores = decoding_data["mean"]

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(
        mean_scores,
        extent=times[[0, -1, 0, -1]],
        interpolation="nearest",
        origin="lower",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
    )

    # Indicate time point zero
    if times.min() < 0 < times.max():
        ax.axvline(0, ls="--", lw=0.5, color="black")
        ax.axhline(0, ls="--", lw=0.5, color="black")

    # Indicate diagonal
    ax.plot(times[[0, -1]], times[[0, -1]], ls="--", lw=0.5, color="black")

    # Axis labels
    _label_time_by_time(
        ax,
        xlabel="Testing time (s)",
        ylabel="Training time (s)",
        decim=decim,
    )

    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    if metric == "roc_auc":
        metric = "ROC AUC"
    cbar.set_label(f"Score ({metric})")

    return fig


def _gen_empty_report(
    *, cfg: SimpleNamespace, subject: str, session: str | None
) -> mne.Report:
    title = f"sub-{subject}"
    if session is not None:
        title += f", ses-{session}"

    report = mne.Report(title=title, raw_psd=True, verbose=False)
    return report


def _contrasts_to_names(contrasts: list[list[str]]) -> list[str]:
    return [f"{c[0]} vs.\n{c[1]}" for c in contrasts]


def add_event_counts(
    *,
    cfg: SimpleNamespace,
    subject: str | None,
    session: str | None,
    task: str | None,
    report: mne.Report,
) -> None:
    try:
        df_events = count_events(BIDSPath(root=cfg.bids_root, session=session))
    except ValueError:
        msg = "Could not read events."
        logger.warning(**gen_log_kwargs(message=msg))
        return
    logger.info(**gen_log_kwargs(message="Adding event counts to report …"))

    if df_events is not None:
        df_events.reset_index(drop=False, inplace=True, col_level=1)
        report.add_html(
            _df_bootstrap_table(df=df_events, data_id="events"),
            title="Event counts",
            tags=("events",),
            replace=True,
        )


def _finalize(
    *,
    report: mne.Report,
    exec_params: SimpleNamespace,
    # passed so logging magic can occur:
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> None:
    """Add system information and the pipeline configuration to the report."""
    # ensure they are always appended
    titles = ["Configuration file", "System information"]
    for title in titles:
        report.remove(title=title, remove_all=True)
    # Print this exactly once
    if _cached_sys_info.cache_info()[-1] == 0:  # never run
        msg = "Adding config and sys info to report"
        logger.info(**gen_log_kwargs(message=msg))
    # No longer need replace=True in these
    report.add_code(
        code=exec_params.config_path,
        title=titles[0],
        tags=("configuration",),
    )
    # We don't use report.add_sys_info so we can use our own cached version
    tags = ("mne-sysinfo",)
    info = _cached_sys_info()
    report.add_code(code=info, title=titles[1], language="shell", tags=tags)
    # Make our code sections take 50% of screen height
    css = """
div.accordion-body pre.my-0 code {
    overflow-y: auto;
    max-height: 50vh;
 }
"""
    if css not in report.include:
        report.add_custom_css(css=css)


# We make a lot of calls to this function and it takes > 1 sec generally
# to run, so run it just once (it shouldn't meaningfully change anyway)
@lru_cache(maxsize=1)
def _cached_sys_info() -> str:
    with StringIO() as f:
        mne.sys_info(f)
        return f.getvalue()


def _all_conditions(*, cfg: SimpleNamespace, task: str | None) -> list[str]:
    if isinstance(cfg.conditions, dict):
        conditions_dict: dict[str, str]
        # Need to inspect if it's nested or not
        for key, val in cfg.conditions.items():
            if isinstance(val, str):
                conditions_dict = cfg.conditions
                break
        else:
            conditions_dict = cfg.conditions[task]
        conditions = list(conditions_dict)
    else:
        conditions = list(cfg.conditions)
    all_contrasts = _get_task_contrasts(contrasts=cfg.contrasts, task=task)
    conditions.extend([contrast["name"] for contrast in all_contrasts])
    return conditions


def _sanitize_cond_tag(cond: str) -> str:
    return str(cond).replace("'", "").replace('"', "").replace(" ", "-")


def _get_prefix_tags(
    *,
    cfg: SimpleNamespace,
    task: str | None,
    run: str | None = None,
    condition: str | None = None,
    contrast: tuple[str, str] | None = None,
    add_contrast: bool = False,
) -> tuple[str, tuple[str, ...]]:
    prefixes = []
    tags: tuple[str, ...] = ()
    if task is not None and len(cfg.all_tasks) > 1:
        prefixes.append(f"task-{task}")
        tags += (f"task-{task}",)
    if run is not None:
        prefixes.append(f"run-{run}")
        tags += (f"run-{run}",)
    if condition is not None:
        condition = _sanitize_cond_tag(condition)
        prefixes += (condition,)
        tags += (condition,)
    if contrast is not None:
        cond_1 = _sanitize_cond_tag(contrast[0])
        cond_2 = _sanitize_cond_tag(contrast[1])
        tags += (f"{cond_1}–{cond_2}",)
        if add_contrast:
            prefixes.append(f"{cond_1} vs. {cond_2}")
    prefix = " ".join(prefixes)
    if prefix:
        prefix = f": {prefix}"
    return prefix, tags


def _imshow_tf(
    vals: FloatArrayT,
    ax: matplotlib.axes.Axes,
    *,
    tmin: FloatArrayT,
    tmax: FloatArrayT,
    fmin: FloatArrayT,
    fmax: FloatArrayT,
    vmin: float,
    vmax: float,
    cmap: str = "RdBu_r",
    mask: FloatArrayT | None = None,
    cmap_masked: Any | None = None,
) -> matplotlib.image.AxesImage:
    """Plot CSP TF decoding scores."""
    # XXX Add support for more metrics
    assert len(vals) == len(tmin) == len(tmax) == len(fmin) == len(fmax)
    mask = np.zeros(vals.shape, dtype=bool) if mask is None else mask
    assert len(vals) == len(mask)
    assert vals.ndim == mask.ndim == 1
    img = None
    for v, t1, t2, f1, f2, m in zip(vals, tmin, tmax, fmin, fmax, mask):
        use_cmap = cmap_masked if m else cmap
        img = ax.imshow(
            np.array([[v]], float),
            cmap=use_cmap,
            extent=[t1, t2, f1, f2],
            aspect="auto",
            interpolation="none",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
    return img


def add_csp_grand_average(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    report: mne.Report,
    cond_1: str,
    cond_2: str,
    fname_csp_freq_results: BIDSPath,
    fname_csp_cluster_results: pd.DataFrame | None,
) -> None:
    """Add CSP decoding results to the grand average report."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    # First, plot decoding scores across frequency bins (entire epochs).
    section = f"Decoding: CSP, N = {len(cfg.subjects)}"
    freq_name_to_bins_map, _ = _handle_csp_args(
        cfg.decoding_csp_times,
        cfg.decoding_csp_freqs,
        cfg.decoding_metric,
        epochs_tmin=cfg.epochs_tmin,
        epochs_tmax=cfg.epochs_tmax,
        time_frequency_freq_min=cfg.time_frequency_freq_min,
        time_frequency_freq_max=cfg.time_frequency_freq_max,
    )

    freq_bin_starts = list()
    freq_bin_widths = list()
    decoding_scores = list()
    error_bars_list = list()
    csp_freq_results = pd.read_excel(fname_csp_freq_results, sheet_name="CSP Frequency")
    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        results = csp_freq_results.loc[
            csp_freq_results["freq_range_name"] == freq_range_name, :
        ]
        results.reset_index(drop=True, inplace=True)
        assert len(results) == len(freq_bins)
        for bi, freq_bin in enumerate(freq_bins):
            freq_bin_starts.append(freq_bin[0])
            freq_bin_widths.append(np.diff(freq_bin)[0])
            decoding_scores.append(results["mean"][bi])
            cis_lower = results["mean_ci_lower"][bi]
            cis_upper = results["mean_ci_upper"][bi]
            error_bars_lower = decoding_scores[-1] - cis_lower
            error_bars_upper = cis_upper - decoding_scores[-1]
            error_bars_list.append(np.stack([error_bars_lower, error_bars_upper]))
            assert len(error_bars_list[-1]) == 2  # lower, upper
            del cis_lower, cis_upper, error_bars_lower, error_bars_upper
    error_bars = np.array(error_bars_list, float).T

    if cfg.decoding_metric == "roc_auc":
        metric = "ROC AUC"

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(
        x=freq_bin_starts,
        width=freq_bin_widths,
        height=decoding_scores,
        align="edge",
        yerr=error_bars,
        edgecolor="black",
    )
    ax.set_ylim([0, 1.02])
    offset = matplotlib.transforms.offset_copy(ax.transData, fig, 0, 5, units="points")
    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        start = freq_bins[0][0]
        stop = freq_bins[-1][1]
        width = stop - start
        ax.text(
            x=start + width / 2,
            y=0.0,
            transform=offset,
            s=freq_range_name,
            ha="center",
            va="bottom",
        )
    ax.axhline(0.5, color="black", linestyle="--", label="chance")
    ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Mean decoding score ({metric})")
    prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=task)
    tags: tuple[str, ...] = (
        "epochs",
        "contrast",
        "decoding",
        "csp",
        f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
    ) + extra_tags
    title = f"CSP decoding{prefix}{cond_1} vs. {cond_2}"
    report.add_figure(
        fig=fig,
        title=title,
        section=section,
        caption="Mean decoding scores. Error bars represent "
        "bootstrapped 95% confidence intervals.",
        tags=tags,
        replace=True,
    )

    # Now, plot decoding scores across time-frequency bins.
    if fname_csp_cluster_results is None:
        return
    csp_cluster_results = loadmat(fname_csp_cluster_results)
    fig, ax = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=True, constrained_layout=True
    )
    n_clu = 0
    cbar = None
    lims = [np.inf, -np.inf, np.inf, -np.inf]
    for freq_range_name, bins in freq_name_to_bins_map.items():
        results = csp_cluster_results[freq_range_name][0][0]
        mean_crossval_scores = results["mean_crossval_scores"].ravel()
        # t_vals = results['t_vals']
        clusters = results["clusters"]
        cluster_p_vals = np.atleast_1d(results["cluster_p_vals"].squeeze())
        tmin = results["time_bin_edges"].ravel()
        tmin, tmax = tmin[:-1], tmin[1:]
        fmin = results["freq_bin_edges"].ravel()
        fmin, fmax = fmin[:-1], fmin[1:]
        lims[0] = min(lims[0], tmin.min())
        lims[1] = max(lims[1], tmax.max())
        lims[2] = min(lims[2], fmin.min())
        lims[3] = max(lims[3], fmax.max())
        # replicate, matching time-frequency order during clustering
        fmin, fmax = np.tile(fmin, len(tmin)), np.tile(fmax, len(tmax))
        tmin, tmax = np.repeat(tmin, len(bins)), np.repeat(tmax, len(bins))
        assert fmin.shape == fmax.shape == tmin.shape == tmax.shape
        assert fmin.shape == mean_crossval_scores.shape
        cluster_t_threshold = results["cluster_t_threshold"].ravel().item()

        significant_cluster_idx = np.where(
            cluster_p_vals < cfg.cluster_permutation_p_threshold
        )[0]
        significant_clusters = clusters[significant_cluster_idx]
        n_clu += len(significant_cluster_idx)

        # XXX Add support for more metrics
        assert cfg.decoding_metric == "roc_auc"
        metric = "ROC AUC"
        vmax = (
            max(
                np.abs(mean_crossval_scores.min() - 0.5),
                np.abs(mean_crossval_scores.max() - 0.5),
            )
            + 0.5
        )
        vmin = 0.5 - (vmax - 0.5)
        # For diverging gray colormap, we need to combine two existing
        # colormaps, as there is no diverging colormap with gray/black at
        # both endpoints.
        from matplotlib.cm import gray, gray_r
        from matplotlib.colors import ListedColormap

        black_to_white = gray(np.linspace(start=0, stop=1, endpoint=False, num=128))
        white_to_black = gray_r(np.linspace(start=0, stop=1, endpoint=False, num=128))
        black_to_white_to_black = np.vstack((black_to_white, white_to_black))
        diverging_gray_cmap = ListedColormap(
            black_to_white_to_black, name="DivergingGray"
        )
        cmap_gray = diverging_gray_cmap
        img = _imshow_tf(
            mean_crossval_scores,
            ax[0],
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            vmin=vmin,
            vmax=vmax,
        )
        if cbar is None:
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Frequency (Hz)")
            ax[1].set_xlabel("Time (s)")
            cbar = fig.colorbar(
                ax=ax[1], shrink=0.75, orientation="vertical", mappable=img
            )
            cbar.set_label(f"Mean decoding score ({metric})")
        offset = matplotlib.transforms.offset_copy(
            ax[0].transData, fig, 6, 0, units="points"
        )
        ax[0].text(
            tmin.min(),
            0.5 * fmin.min() + 0.5 * fmax.max(),
            freq_range_name,
            transform=offset,
            ha="left",
            va="center",
            rotation=90,
        )

        if len(significant_clusters):
            # Create a masked array that only shows the T-values for
            # time-frequency bins that belong to significant clusters.
            if len(significant_clusters) == 1:
                mask = ~significant_clusters[0].astype(bool)
            else:
                mask = ~np.logical_or(*significant_clusters)
            mask = mask.ravel()
        else:
            mask = np.ones(mean_crossval_scores.shape, dtype=bool)
        _imshow_tf(
            mean_crossval_scores,
            ax[1],
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            cmap_masked=cmap_gray,
        )

    ax[0].set_xlim(lims[:2])
    ax[0].set_ylim(lims[2:])
    ax[0].set_title("Scores")
    ax[1].set_title("Masked")
    tags = (
        "epochs",
        "contrast",
        "decoding",
        "csp",
        f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
    ) + extra_tags
    title = f"CSP TF decoding{prefix}{cond_1} vs. {cond_2}"
    report.add_figure(
        fig=fig,
        title=title,
        section=section,
        caption=f"Found {n_clu} "
        f"cluster{_pl(n_clu)} with "
        f"p < {cfg.cluster_permutation_p_threshold} "
        f"(clustering bins with absolute t-values > "
        f"{round(cluster_t_threshold, 3)}).",
        tags=tags,
        replace=True,
    )


@contextlib.contextmanager
def _agg_backend() -> Generator[None, None, None]:
    import matplotlib

    backend = matplotlib.get_backend()
    matplotlib.use("agg", force=True)
    try:
        yield
    finally:
        if backend.lower() != "agg":
            import matplotlib.pyplot as plt

            plt.close("all")
            matplotlib.use(backend, force=True)


def _add_raw(
    *,
    cfg: SimpleNamespace,
    report: mne.report.Report,
    bids_path_in: BIDSPath,
    raw: BaseRaw,
    title_prefix: str,
    tags: tuple[str, ...] = (),
    extra_html: str | None = None,
) -> None:
    prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=bids_path_in.task)
    title = f"{title_prefix}{prefix}"
    plot_raw_psd = (
        cfg.plot_psd_for_runs == "all"
        or bids_path_in.run in cfg.plot_psd_for_runs
        or bids_path_in.task in cfg.plot_psd_for_runs
    )
    tags = ("raw",) + tags + extra_tags
    with mne.use_log_level("error"):
        report.add_raw(
            raw=raw,
            title=title,
            butterfly=5,
            psd=plot_raw_psd,
            tags=tags,
            # caption=bids_path_in.basename,  # TODO upstream
            replace=True,
        )
        if extra_html is not None:
            report.add_html(
                extra_html,
                title=title,
                tags=tags,
                section=title,
                replace=True,
            )


def _render_bem(
    *,
    cfg: SimpleNamespace,
    report: mne.report.Report,
    subject: str,
    session: str | None,
) -> None:
    logger.info(**gen_log_kwargs(message="Rendering MRI slices with BEM contours."))
    report.add_bem(
        subject=cfg.fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        title="BEM",
        width=256,
        decim=8,
        replace=True,
        n_jobs=1,  # prevent automatic parallelization
    )
