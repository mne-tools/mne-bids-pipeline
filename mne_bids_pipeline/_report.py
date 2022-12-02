import contextlib
from functools import lru_cache
from io import StringIO
import os.path as op
from pathlib import Path
from typing import Optional, List, Literal
from types import SimpleNamespace

from filelock import FileLock
import matplotlib.transforms
import numpy as np
import pandas as pd
from scipy.io import loadmat

import mne
from mne.utils import _pl
from mne_bids import BIDSPath
from mne_bids.stats import count_events

from ._config_utils import (
    sanitize_cond_name, get_subjects, _restrict_analyze_channels)
from ._decoding import _handle_csp_args
from ._logging import logger, gen_log_kwargs


@contextlib.contextmanager
def _open_report(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str] = None,
):
    fname_report = BIDSPath(
        subject=subject,
        session=session,
        # Report is across all runs, but for logging purposes it's helpful
        # to pass the run for gen_log_kwargs
        run=None,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        extension='.h5',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        suffix='report',
        check=False
    ).fpath
    # prevent parallel file access
    with FileLock(f'{fname_report}.lock'), _agg_backend():
        if not fname_report.is_file():
            msg = 'Initializing report HDF5 file'
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
                f'Could not open report HDF5 file:\n{fname_report}\n'
                f'Got error:\n{exc}\nPerhaps you need to delete it?') from None
        try:
            yield report
        finally:
            try:
                msg = 'Adding config and sys info to report'
                logger.info(**gen_log_kwargs(message=msg))
                _finalize(
                    report=report,
                    exec_params=exec_params,
                    subject=subject,
                    session=session,
                    run=run,
                )
            except Exception:
                pass
            fname_report_html = fname_report.with_suffix('.html')
            msg = f'Saving report: {fname_report_html}'
            logger.info(**gen_log_kwargs(message=msg))
            report.save(fname_report, overwrite=True)
            report.save(
                fname_report_html, overwrite=True,
                open_browser=False)


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
    contrast_names: List[str],
    scores: List[np.ndarray],
    metric: str,
    kind: Literal['single-subject', 'grand-average'] = 'single-subject',
):
    """Plot cross-validation results from full-epochs decoding.
    """
    import matplotlib.pyplot as plt  # nested import to help joblib
    import seaborn as sns

    if metric == 'roc_auc':
        metric = 'ROC AUC'
    score_label = f'Score ({metric})'

    data = pd.DataFrame({
        'Contrast': np.array([
            [c] * len(scores[0])
            for c in contrast_names
        ]).flatten(),
        score_label: np.hstack(scores),
    })

    if kind == 'grand-average':
        # First create a grid of boxplots …
        g = sns.catplot(
            data=data, y=score_label, kind='box',
            col='Contrast', col_wrap=3, aspect=0.33
        )
        # … and now add swarmplots on top to visualize every single data point.
        g.map_dataframe(sns.swarmplot, y=score_label, color='black')
        caption = (
            f'Based on N={len(scores[0])} '
            f'subjects. Each dot represents the mean cross-validation score '
            f'for a single subject. The dashed line is expected chance '
            f'performance.'
        )
    else:
        # First create a grid of swarmplots to visualize every single
        # cross-validation score.
        g = sns.catplot(
            data=data, y=score_label, kind='swarm',
            col='Contrast', col_wrap=3, aspect=0.33, color='black'
        )

        # And now add the mean CV score on top.
        def _plot_mean_cv_score(x, **kwargs):
            plt.plot(x.mean(), **kwargs)

        g.map(
            _plot_mean_cv_score, score_label, marker='+', color='red',
            ms=15, label='mean score', zorder=99
        )
        caption = (
            'Each black dot represents the single cross-validation score. '
            f'The red cross is the mean of all {len(scores[0])} '
            'cross-validation scores. '
            'The dashed line is expected chance performance.'
        )
        plt.xlim([-0.1, 0.1])

    g.map(plt.axhline, y=0.5, ls='--', lw=0.5, color='black', zorder=99)
    g.set_titles('{col_name}')  # use this argument literally!
    g.set_xlabels('')

    fig = g.fig
    return fig, caption


def _plot_time_by_time_decoding_scores(
    *,
    times: np.ndarray,
    cross_val_scores: np.ndarray,
    metric: str,
    time_generalization: bool,
    decim: int,
):
    """Plot cross-validation results from time-by-time decoding.
    """
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
    ax.axhline(0.5, ls='--', lw=0.5, color='black', label='chance')
    if times.min() < 0 < times.max():
        ax.axvline(0, ls='-', lw=0.5, color='black')
    ax.fill_between(x=times, y1=min_scores, y2=max_scores, color='lightgray',
                    alpha=0.5, label='range [min, max]')
    ax.plot(times, mean_scores, ls='-', lw=2, color='black',
            label='mean')

    _label_time_by_time(ax, xlabel='Time (s)', decim=decim)
    if metric == 'roc_auc':
        metric = 'ROC AUC'
    ax.set_ylabel(f'Score ({metric})')
    ax.set_ylim((-0.025, 1.025))
    ax.legend(loc='lower right')

    return fig


def _label_time_by_time(ax, *, decim, xlabel=None, ylabel=None):
    extra = ''
    if decim > 1:
        extra = f' (decim={decim})'
    if xlabel is not None:
        ax.set_xlabel(f'{xlabel}{extra}')
    if ylabel is not None:
        ax.set_ylabel(f'{ylabel}{extra}')


def _plot_time_by_time_decoding_scores_gavg(*, cfg, decoding_data):
    """Plot the grand-averaged decoding scores.
    """
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    times = decoding_data['times'].squeeze()
    mean_scores = decoding_data['mean'].squeeze()
    se_lower = mean_scores - decoding_data['mean_se'].squeeze()
    se_upper = mean_scores + decoding_data['mean_se'].squeeze()
    ci_lower = decoding_data['mean_ci_lower'].squeeze()
    ci_upper = decoding_data['mean_ci_upper'].squeeze()
    decim = decoding_data['decim'].item()

    if cfg.decoding_time_generalization:
        # Only use the diagonal values (classifiers trained and tested on the
        # same time points).
        mean_scores = np.diag(mean_scores)
        se_lower = np.diag(se_lower)
        se_upper = np.diag(se_upper)
        ci_lower = np.diag(ci_lower)
        ci_upper = np.diag(ci_upper)

    metric = cfg.decoding_metric
    clusters = np.atleast_1d(decoding_data['clusters'].squeeze())

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_ylim((-0.025, 1.025))

    # Start with plotting the significant time periods according to the
    # cluster-based permutation test
    n_significant_clusters_plotted = 0
    for cluster in clusters:
        cluster_times = np.atleast_1d(cluster['times'][0][0].squeeze())
        cluster_p = cluster['p_value'][0][0].item()
        if cluster_p >= cfg.cluster_permutation_p_threshold:
            continue

        # Only add the label once
        if n_significant_clusters_plotted == 0:
            label = (f'$p$ < {cfg.cluster_permutation_p_threshold} '
                     f'(cluster pemutation)')
        else:
            label = None

        ax.fill_betweenx(
            y=ax.get_ylim(),
            x1=cluster_times[0],
            x2=cluster_times[-1],
            facecolor='orange',
            alpha=0.15,
            label=label
        )
        n_significant_clusters_plotted += 1

    ax.axhline(0.5, ls='--', lw=0.5, color='black', label='chance')
    if times.min() < 0 < times.max():
        ax.axvline(0, ls='-', lw=0.5, color='black')
    ax.fill_between(x=times, y1=ci_lower, y2=ci_upper, color='lightgray',
                    alpha=0.5, label='95% confidence interval')

    ax.plot(times, mean_scores, ls='-', lw=2, color='black',
            label='mean')
    ax.plot(times, se_lower, ls='-.', lw=0.5, color='gray',
            label='mean ± standard error')
    ax.plot(times, se_upper, ls='-.', lw=0.5, color='gray')
    ax.text(0.05, 0.05, s=f'$N$={decoding_data["N"].squeeze()}',
            fontsize='x-large', horizontalalignment='left',
            verticalalignment='bottom', transform=ax.transAxes)

    _label_time_by_time(ax, xlabel='Time (s)', decim=decim)
    if metric == 'roc_auc':
        metric = 'ROC AUC'
    ax.set_ylabel(f'Score ({metric})')
    ax.legend(loc='lower right')

    return fig


def plot_time_by_time_decoding_t_values(decoding_data):
    """Plot the t-values used to form clusters for the permutation test.
    """
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    all_times = decoding_data['cluster_all_times'].squeeze()
    all_t_values = decoding_data['cluster_all_t_values'].squeeze()
    t_threshold = decoding_data['cluster_t_threshold']
    decim = decoding_data['decim']

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(all_times, all_t_values, ls='-', color='black',
            label='observed $t$-values')
    ax.axhline(t_threshold, ls='--', color='red', label='threshold')

    ax.text(0.05, 0.05, s=f'$N$={decoding_data["N"].squeeze()}',
            fontsize='x-large', horizontalalignment='left',
            verticalalignment='bottom', transform=ax.transAxes)

    _label_time_by_time(ax, xlabel='Time (s)', decim=decim)
    ax.set_ylabel('$t$-value')
    ax.legend(loc='lower right')

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
    decoding_data,
    metric: str,
    kind: Literal['single-subject', 'grand-average']
):
    """Plot time generalization matrix.
    """
    import matplotlib.pyplot as plt  # nested import to help joblib

    # We squeeze() to make Matplotlib happy.
    times = decoding_data['times'].squeeze()
    decim = decoding_data['decim'].item()
    if kind == 'single-subject':
        # take the mean across CV scores
        mean_scores = decoding_data['scores'].mean(axis=0)
    else:
        mean_scores = decoding_data['mean']

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(
        mean_scores,
        extent=times[[0, -1, 0, -1]],
        interpolation='nearest',
        origin='lower',
        cmap='RdBu_r',
        vmin=0,
        vmax=1
    )

    # Indicate time point zero
    if times.min() < 0 < times.max():
        ax.axvline(0, ls='--', lw=0.5, color='black')
        ax.axhline(0, ls='--', lw=0.5, color='black')

    # Indicate diagonal
    ax.plot(times[[0, -1]], times[[0, -1]], ls='--', lw=0.5, color='black')

    # Axis labels
    _label_time_by_time(
        ax,
        xlabel='Testing time (s)',
        ylabel='Training time (s)',
        decim=decim,
    )

    # Color bar
    cbar = plt.colorbar(im, ax=ax)
    if metric == 'roc_auc':
        metric = 'ROC AUC'
    cbar.set_label(f'Score ({metric})')

    return fig


def _gen_empty_report(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str]
) -> mne.Report:
    title = f'sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    report = mne.Report(title=title, raw_psd=True)
    return report


def _contrasts_to_names(contrasts: List[List[str]]) -> List[str]:
    return [f'{c[0]} vs.\n{c[1]}' for c in contrasts]


def add_event_counts(*,
                     cfg,
                     subject: Optional[str],
                     session: Optional[str],
                     report: mne.Report) -> None:
    try:
        df_events = count_events(BIDSPath(root=cfg.bids_root,
                                          session=session))
    except ValueError:
        msg = 'Could not read events.'
        logger.warning(**gen_log_kwargs(message=msg))
        df_events = None

    if df_events is not None:
        css_classes = ('table', 'table-striped', 'table-borderless',
                       'table-hover')
        report.add_html(
            f'<div class="event-counts">\n'
            f'{df_events.to_html(classes=css_classes, border=0)}\n'
            f'</div>',
            title='Event counts',
            tags=('events',),
            replace=True,
        )
        css = ('.event-counts {\n'
               '  display: -webkit-box;\n'
               '  display: -ms-flexbox;\n'
               '  display: -webkit-flex;\n'
               '  display: flex;\n'
               '  justify-content: center;\n'
               '  text-align: center;\n'
               '}\n\n'
               'th, td {\n'
               '  text-align: center;\n'
               '}\n')
        if css not in report.include:
            report.add_custom_css(css=css)


def _finalize(
    *,
    report: mne.Report,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
) -> None:
    """Add system information and the pipeline configuration to the report."""
    # ensure they are always appended
    titles = ['Configuration file', 'System information']
    for title in titles:
        report.remove(title=title, remove_all=True)
    # No longer need replace=True in these
    report.add_code(
        code=exec_params.config_path,
        title=titles[0],
        tags=('configuration',),
    )
    # We don't use report.add_sys_info so we can use our own cached version
    tags = ('mne-sysinfo',)
    info = _cached_sys_info()
    report.add_code(code=info, title=titles[1], language='shell', tags=tags)
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
def _cached_sys_info(f):
    with StringIO() as f:
        mne.sys_info(f)
        return f.getvalue()


def _all_conditions(*, cfg):
    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()
    conditions.extend([contrast["name"] for contrast in cfg.all_contrasts])
    return conditions


def run_report_average_sensor(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> None:
    msg = 'Generating grand average report …'
    logger.info(**gen_log_kwargs(message=msg))
    assert matplotlib.get_backend() == 'agg', matplotlib.get_backend()

    evoked_fname = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='ave',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    title = f'sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    all_evokeds = mne.read_evokeds(evoked_fname)
    for evoked in all_evokeds:
        _restrict_analyze_channels(evoked, cfg)
    conditions = _all_conditions(cfg=cfg)
    assert len(conditions) == len(all_evokeds)
    all_evokeds = {
        cond: evoked
        for cond, evoked in zip(conditions, all_evokeds)
    }

    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:

        #######################################################################
        #
        # Add event stats.
        #
        add_event_counts(
            cfg=cfg,
            report=report,
            subject=subject,
            session=session,
        )

        #######################################################################
        #
        # Visualize evoked responses.
        #
        if all_evokeds:
            msg = (
                f'Adding {len(all_evokeds)} evoked signals and contrasts to '
                'the report.'
            )
        else:
            msg = 'No evoked conditions or contrasts found.'
        logger.info(**gen_log_kwargs(message=msg))
        for condition, evoked in all_evokeds.items():
            tags = ('evoked', _sanitize_cond_tag(condition))
            if condition in cfg.conditions:
                title = f'Condition: {condition}'
            else:  # It's a contrast of two conditions.
                title = f'Contrast: {condition}'
                tags = tags + ('contrast',)

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

        #######################################################################
        #
        # Visualize decoding results.
        #
        if cfg.decode and cfg.decoding_contrasts:
            msg = 'Adding decoding results.'
            logger.info(**gen_log_kwargs(message=msg))
            add_decoding_grand_average(
                session=session, cfg=cfg, report=report
            )

        if cfg.decode and cfg.decoding_csp:
            # No need for a separate message here because these are very quick
            # and the general message above is sufficient
            add_csp_grand_average(
                session=session, cfg=cfg, report=report
            )


def run_report_average_source(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> None:
    #######################################################################
    #
    # Visualize forward solution, inverse operator, and inverse solutions.
    #
    evoked_fname = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='ave',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )
    evokeds = mne.read_evokeds(evoked_fname)
    method = cfg.inverse_method
    inverse_str = method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph2fsaverage'
    conditions = _all_conditions(cfg=cfg)
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        for condition, evoked in zip(conditions, evokeds):
            tags = (
                'source-estimate',
                _sanitize_cond_tag(condition),
            )
            if condition in cfg.conditions:
                title = f'Average: {condition}'
            else:  # It's a contrast of two conditions.
                title = f'Average contrast: {condition}'
                tags = tags + ('contrast',)
            cond_str = sanitize_cond_name(condition)
            fname_stc_avg = evoked_fname.copy().update(
                suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}',
                extension=None)
            if not Path(f'{fname_stc_avg.fpath}-lh.stc').exists():
                continue
            report.add_stc(
                stc=fname_stc_avg,
                title=title,
                subject='fsaverage',
                subjects_dir=cfg.fs_subjects_dir,
                n_time_points=cfg.report_stc_n_time_points,
                tags=tags,
                replace=True,
            )


def add_decoding_grand_average(
    *,
    session: Optional[str],
    cfg: SimpleNamespace,
    report: mne.Report,
):
    """Add decoding results to the grand average report."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    bids_path = BIDSPath(
        subject='average',
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='ave',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    # Full-epochs decoding
    all_decoding_scores = []
    for contrast in cfg.decoding_contrasts:
        cond_1, cond_2 = contrast
        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        processing = f'{a_vs_b}+FullEpochs+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')
        fname_decoding = bids_path.copy().update(
            processing=processing,
            suffix='decoding',
            extension='.mat'
        )
        decoding_data = loadmat(fname_decoding)
        all_decoding_scores.append(
            np.atleast_1d(decoding_data['scores'].squeeze())
        )
        del fname_decoding, processing, a_vs_b, decoding_data

    fig, caption = _plot_full_epochs_decoding_scores(
        contrast_names=_contrasts_to_names(cfg.decoding_contrasts),
        scores=all_decoding_scores,
        metric=cfg.decoding_metric,
        kind='grand-average'
    )
    title = f'Full-epochs decoding: {cond_1} vs. {cond_2}'
    report.add_figure(
        fig=fig,
        title=title,
        section='Decoding: full-epochs',
        caption=caption,
        tags=(
            'epochs',
            'contrast',
            'decoding',
            *[f'{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}'
              for cond_1, cond_2 in cfg.decoding_contrasts]
        ),
        replace=True,
    )
    # close figure to save memory
    plt.close(fig)
    del fig, caption, title

    # Time-by-time decoding
    for contrast in cfg.decoding_contrasts:
        cond_1, cond_2 = contrast
        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        section = 'Decoding: time-by-time'
        tags = (
            'epochs',
            'contrast',
            'decoding',
            f'{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}'
        )
        processing = f'{a_vs_b}+TimeByTime+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')
        fname_decoding = bids_path.copy().update(
            processing=processing,
            suffix='decoding',
            extension='.mat'
        )
        decoding_data = loadmat(fname_decoding)
        del fname_decoding, processing, a_vs_b

        # Plot scores
        fig = _plot_time_by_time_decoding_scores_gavg(
            cfg=cfg,
            decoding_data=decoding_data,
        )
        caption = (
            f'Based on N={decoding_data["N"].squeeze()} '
            f'subjects. Standard error and confidence interval '
            f'of the mean were bootstrapped with {cfg.n_boot} '
            f'resamples. CI must not be used for statistical inference here, '
            f'as it is not corrected for multiple testing.'
        )
        if len(get_subjects(cfg)) > 1:
            caption += (
                f' Time periods with decoding performance significantly above '
                f'chance, if any, were derived with a one-tailed '
                f'cluster-based permutation test '
                f'({decoding_data["cluster_n_permutations"].squeeze()} '
                f'permutations) and are highlighted in yellow.'
            )
            title = f'Decoding over time: {cond_1} vs. {cond_2}'
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
            fig = plot_time_by_time_decoding_t_values(
                decoding_data=decoding_data
            )
            t_threshold = np.round(
                decoding_data['cluster_t_threshold'],
                3
            ).item()
            caption = (
                f'Observed t-values. Time points with '
                f't-values > {t_threshold} were used to form clusters.'
            )
            report.add_figure(
                fig=fig,
                title=f't-values across time: {cond_1} vs. {cond_2}',
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
                kind='grand-average'
            )
            caption = (
                f'Time generalization (generalization across time, GAT): '
                f'each classifier is trained on each time point, and tested '
                f'on all other time points. The results were averaged across '
                f'N={decoding_data["N"].item()} subjects.'
            )
            title = f'Time generalization: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)


def _sanitize_cond_tag(cond):
    return cond.lower().replace(' ', '-')


def _imshow_tf(vals, ax, *, tmin, tmax, fmin, fmax, vmin, vmax, cmap='RdBu_r',
               mask=None, cmap_masked=None):
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
            cmap=use_cmap, extent=[t1, t2, f1, f2], aspect='auto',
            interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
        )
    return img


def add_csp_grand_average(
    *,
    session: str,
    cfg: SimpleNamespace,
    report: mne.Report,
):
    """Add CSP decoding results to the grand average report."""
    import matplotlib.pyplot as plt  # nested import to help joblib

    bids_path = BIDSPath(
        subject='average',
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='decoding',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    # First, plot decoding scores across frequency bins (entire epochs).
    section = 'Decoding: CSP'
    freq_name_to_bins_map = _handle_csp_args(
        cfg.decoding_csp_times,
        cfg.decoding_csp_freqs,
        cfg.decoding_metric,
    )
    for contrast in cfg.decoding_contrasts:
        cond_1, cond_2 = contrast
        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')
        fname_csp_freq_results = bids_path.copy().update(
            processing=processing,
            extension='.xlsx',
        )
        csp_freq_results = pd.read_excel(
            fname_csp_freq_results,
            sheet_name='CSP Frequency'
        )

        freq_bin_starts = list()
        freq_bin_widths = list()
        decoding_scores = list()
        error_bars = list()
        for freq_range_name, freq_bins in freq_name_to_bins_map.items():
            results = csp_freq_results.loc[
                csp_freq_results['freq_range_name'] == freq_range_name, :
            ]
            results.reset_index(drop=True, inplace=True)
            assert len(results) == len(freq_bins)
            for bi, freq_bin in enumerate(freq_bins):
                freq_bin_starts.append(freq_bin[0])
                freq_bin_widths.append(np.diff(freq_bin)[0])
                decoding_scores.append(results['mean'][bi])
                cis_lower = results['mean_ci_lower'][bi]
                cis_upper = results['mean_ci_upper'][bi]
                error_bars_lower = decoding_scores[-1] - cis_lower
                error_bars_upper = cis_upper - decoding_scores[-1]
                error_bars.append(
                    np.stack([error_bars_lower, error_bars_upper]))
                assert len(error_bars[-1]) == 2  # lower, upper
                del cis_lower, cis_upper, error_bars_lower, error_bars_upper
        error_bars = np.array(error_bars, float).T

        if cfg.decoding_metric == 'roc_auc':
            metric = 'ROC AUC'

        fig, ax = plt.subplots(constrained_layout=True)
        ax.bar(
            x=freq_bin_starts,
            width=freq_bin_widths,
            height=decoding_scores,
            align='edge',
            yerr=error_bars,
            edgecolor='black',
        )
        ax.set_ylim([0, 1.02])
        offset = matplotlib.transforms.offset_copy(
            ax.transData, fig, 0, 5, units='points')
        for freq_range_name, freq_bins in freq_name_to_bins_map.items():
            start = freq_bins[0][0]
            stop = freq_bins[-1][1]
            width = stop - start
            ax.text(
                x=start + width / 2,
                y=0.,
                transform=offset,
                s=freq_range_name,
                ha='center',
                va='bottom',
            )
        ax.axhline(0.5, color='black', linestyle='--', label='chance')
        ax.legend()
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'Mean decoding score ({metric})')
        tags = (
            'epochs',
            'contrast',
            'decoding',
            'csp',
            f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
        )
        title = f'CSP decoding: {cond_1} vs. {cond_2}'
        report.add_figure(
            fig=fig,
            title=title,
            section=section,
            caption='Mean decoding scores. Error bars represent '
                    'bootstrapped 95% confidence intervals.',
            tags=tags,
            replace=True,
        )

    # Now, plot decoding scores across time-frequency bins.
    for contrast in cfg.decoding_contrasts:
        cond_1, cond_2 = contrast
        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')
        fname_csp_cluster_results = bids_path.copy().update(
            processing=processing,
            extension='.mat',
        )
        csp_cluster_results = loadmat(fname_csp_cluster_results)

        fig, ax = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True,
            constrained_layout=True)
        n_clu = 0
        cbar = None
        lims = [np.inf, -np.inf, np.inf, -np.inf]
        for freq_range_name, bins in freq_name_to_bins_map.items():
            results = csp_cluster_results[freq_range_name][0][0]
            mean_crossval_scores = results['mean_crossval_scores'].ravel()
            # t_vals = results['t_vals']
            clusters = results['clusters']
            cluster_p_vals = np.atleast_1d(results['cluster_p_vals'].squeeze())
            tmin = results['time_bin_edges'].ravel()
            tmin, tmax = tmin[:-1], tmin[1:]
            fmin = results['freq_bin_edges'].ravel()
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
            cluster_t_threshold = results['cluster_t_threshold'].ravel().item()

            significant_cluster_idx = np.where(
                cluster_p_vals < cfg.cluster_permutation_p_threshold
            )[0]
            significant_clusters = clusters[significant_cluster_idx]
            n_clu += len(significant_cluster_idx)

            # XXX Add support for more metrics
            assert cfg.decoding_metric == 'roc_auc'
            metric = 'ROC AUC'
            vmax = max(
                np.abs(mean_crossval_scores.min() - 0.5),
                np.abs(mean_crossval_scores.max() - 0.5)
            ) + 0.5
            vmin = 0.5 - (vmax - 0.5)
            # For diverging gray colormap, we need to combine two existing
            # colormaps, as there is no diverging colormap with gray/black at
            # both endpoints.
            from matplotlib.cm import gray, gray_r
            from matplotlib.colors import ListedColormap

            black_to_white = gray(
                np.linspace(start=0, stop=1, endpoint=False, num=128)
            )
            white_to_black = gray_r(
                np.linspace(start=0, stop=1, endpoint=False, num=128)
            )
            black_to_white_to_black = np.vstack(
                (black_to_white, white_to_black)
            )
            diverging_gray_cmap = ListedColormap(
                black_to_white_to_black, name='DivergingGray'
            )
            cmap_gray = diverging_gray_cmap
            img = _imshow_tf(
                mean_crossval_scores, ax[0],
                tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                vmin=vmin, vmax=vmax)
            if cbar is None:
                ax[0].set_xlabel('Time (s)')
                ax[0].set_ylabel('Frequency (Hz)')
                ax[1].set_xlabel('Time (s)')
                cbar = fig.colorbar(
                    ax=ax[1], shrink=0.75, orientation='vertical',
                    mappable=img)
                cbar.set_label(f'Mean decoding score ({metric})')
            offset = matplotlib.transforms.offset_copy(
                ax[0].transData, fig, 6, 0, units='points')
            ax[0].text(tmin.min(),
                       0.5 * fmin.min() + 0.5 * fmax.max(),
                       freq_range_name, transform=offset,
                       ha='left', va='center', rotation=90)

            if len(significant_clusters):
                # Create a masked array that only shows the T-values for
                # time-frequency bins that belong to significant clusters.
                if len(significant_clusters) == 1:
                    mask = ~significant_clusters[0].astype(bool)
                else:
                    mask = ~np.logical_or(
                        *significant_clusters
                    )
                mask = mask.ravel()
            else:
                mask = np.ones(mean_crossval_scores.shape, dtype=bool)
            _imshow_tf(
                mean_crossval_scores, ax[1],
                tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                vmin=vmin, vmax=vmax, mask=mask, cmap_masked=cmap_gray)

        ax[0].set_xlim(lims[:2])
        ax[0].set_ylim(lims[2:])
        ax[0].set_title('Scores')
        ax[1].set_title('Masked')
        tags = (
            'epochs',
            'contrast',
            'decoding',
            'csp',
            f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
        )
        title = f'CSP TF decoding: {cond_1} vs. {cond_2}'
        report.add_figure(
            fig=fig,
            title=title,
            section=section,
            caption=f'Found {n_clu} '
                    f'cluster{_pl(n_clu)} with '
                    f'p < {cfg.cluster_permutation_p_threshold} '
                    f'(clustering bins with absolute t-values > '
                    f'{round(cluster_t_threshold, 3)}).',
            tags=tags,
            replace=True,
        )


@contextlib.contextmanager
def _agg_backend():
    import matplotlib
    backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)
    try:
        yield
    finally:
        matplotlib.use(backend, force=True)


def _add_raw(
    *,
    cfg: SimpleNamespace,
    report: mne.report.Report,
    bids_path_in: BIDSPath,
    title: str,
):
    if bids_path_in.run is not None:
        title += f', run {bids_path_in.run}'
    elif bids_path_in.task in ('noise', 'rest'):
        title += f', run {bids_path_in.task}'
    plot_raw_psd = (
        cfg.plot_psd_for_runs == 'all' or
        bids_path_in.run in cfg.plot_psd_for_runs or
        bids_path_in.task in cfg.plot_psd_for_runs
    )
    with mne.use_log_level('error'):
        report.add_raw(
            raw=bids_path_in,
            title=title,
            butterfly=5,
            psd=plot_raw_psd,
            tags=('raw', 'filtered', f'run-{bids_path_in.run}'),
            # caption=bids_path_in.basename,  # TODO upstream
            replace=True,
        )
