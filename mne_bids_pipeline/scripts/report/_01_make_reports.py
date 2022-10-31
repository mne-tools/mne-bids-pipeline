"""Make reports.

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

import contextlib
import os
import os.path as op
from pathlib import Path
from typing import Optional, List, Literal
from types import SimpleNamespace

from scipy.io import loadmat
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
from mne_bids.stats import count_events

from ..._config_utils import (
    get_noise_cov_bids_path, get_subjects, sanitize_cond_name,
    get_task, get_datatype, get_deriv_root, get_sessions,
    _restrict_analyze_channels, get_fs_subjects_dir, get_fs_subject,
    get_runs, get_bids_root, get_decoding_contrasts, get_all_contrasts)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import get_parallel_backend, parallel_func
from ..._run import (
    failsafe_run, save_logs, _update_for_splits, _sanitize_callable)
from ..._reject import _get_reject
from ..._viz import plot_auto_scores


def get_events(cfg, subject, session):
    raws_filt = []
    raw_fname = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         recording=cfg.rec,
                         space=cfg.space,
                         processing='filt',
                         suffix='raw',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    for run in cfg.runs:
        this_raw_fname = raw_fname.copy().update(run=run)
        this_raw_fname = _update_for_splits(this_raw_fname, None, single=True)
        raw_filt = mne.io.read_raw_fif(this_raw_fname)
        raws_filt.append(raw_filt)
        del this_raw_fname

    # Concatenate the filtered raws and extract the events.
    raw_filt_concat = mne.concatenate_raws(raws_filt, on_mismatch='warn')
    events, event_id = mne.events_from_annotations(raw=raw_filt_concat)
    return (events, event_id, raw_filt_concat.info['sfreq'],
            raw_filt_concat.first_samp)


def get_er_path(cfg, subject, session):
    raw_fname = BIDSPath(subject=subject,
                         session=session,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         task='noise',
                         processing='filt',
                         suffix='raw',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)
    raw_fname = _update_for_splits(
        raw_fname, None, single=True, allow_missing=True)
    return raw_fname


def plot_auto_scores_(cfg, subject, session):
    """Plot automated bad channel detection scores.
    """
    import json_tricks

    fname_scores = BIDSPath(subject=subject,
                            session=session,
                            task=cfg.task,
                            acquisition=cfg.acq,
                            run=None,
                            processing=cfg.proc,
                            recording=cfg.rec,
                            space=cfg.space,
                            suffix='scores',
                            extension='.json',
                            datatype=cfg.datatype,
                            root=cfg.deriv_root,
                            check=False)

    all_figs = []
    all_captions = []
    for run in cfg.runs:
        fname_scores.update(run=run)
        auto_scores = json_tricks.loads(
            fname_scores.fpath.read_text(encoding='utf-8-sig')
        )

        figs = plot_auto_scores(auto_scores, ch_types=cfg.ch_types)
        all_figs.extend(figs)

        # Could be more than 1 fig, e.g. "grad" and "mag"
        captions = [f'Run {run}'] * len(figs)
        all_captions.extend(captions)

    assert all_figs
    return all_figs, all_captions


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


def run_report_preprocessing(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    report: Optional[mne.Report]
) -> mne.Report:
    import matplotlib.pyplot as plt  # nested import to help joblib

    msg = 'Generating preprocessing report …'
    logger.info(
        **gen_log_kwargs(message=msg, subject=subject, session=session)
    )

    if report is None:
        report = _gen_empty_report(
            cfg=cfg,
            subject=subject,
            session=session
        )

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    fnames_raw_filt = []
    for run in cfg.runs:
        fname = bids_path.copy().update(
            run=run, processing='filt',
            suffix='raw', check=False
        )
        fname = _update_for_splits(fname, None, single=True)
        fnames_raw_filt.append(fname)

    fname_epo_not_clean = bids_path.copy().update(suffix='epo')
    fname_epo_clean = bids_path.copy().update(processing='clean', suffix='epo')
    fname_ica = bids_path.copy().update(suffix='ica')
    fname_ssp = bids_path.copy().update(suffix='proj')
    fname_eog_epochs = bids_path.copy().update(suffix='eog-epo')
    fname_ecg_epochs = bids_path.copy().update(suffix='ecg-epo')

    for fname in fnames_raw_filt:
        msg = 'Adding filtered raw data to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session, run=fname.run
            )
        )

        title = 'Raw'
        if fname.run is not None:
            title += f', run {fname.run}'

        if (
            cfg.plot_psd_for_runs == 'all' or
            fname.run in cfg.plot_psd_for_runs
        ):
            plot_raw_psd = True
        else:
            plot_raw_psd = False

        report.add_raw(
            raw=fname,
            title=title,
            butterfly=5,
            psd=plot_raw_psd,
            tags=('raw', 'filtered', f'run-{fname.run}'),
            # caption=fname.basename  # TODO upstream
        )
        del plot_raw_psd

    er_path = get_er_path(cfg=cfg, subject=subject, session=session)
    if er_path.fpath.exists():
        msg = 'Adding filtered empty-room raw data to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )

        report.add_raw(
            raw=er_path,
            title='Empty-Room',
            butterfly=5,
            tags=('raw', 'empty-room')
            # caption=er_path.basename  # TODO upstream
        )

    # Visualize automated noisy channel detection.
    if cfg.find_noisy_channels_meg:
        msg = 'Adding visualization of noisy channel detection to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )

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
            tags=tags
        )
        for fig in figs:
            plt.close(fig)

    # Visualize events.
    if not cfg.task_is_rest:
        msg = 'Adding events plot to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )

        events, event_id, sfreq, first_samp = get_events(
            cfg=cfg, subject=subject, session=session
        )
        report.add_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            first_samp=first_samp,
            title='Events',
            # caption='Events in filtered continuous data',  # TODO upstream
        )

    ###########################################################################
    #
    # Visualize uncleaned epochs.
    #
    msg = 'Adding uncleaned epochs to report.'
    logger.info(
        **gen_log_kwargs(
            message=msg, subject=subject, session=session
        )
    )
    epochs = mne.read_epochs(fname_epo_not_clean)
    # Add PSD plots for 30s of data or all epochs if we have less available
    if len(epochs) * (epochs.tmax - epochs.tmin) < 30:
        psd = True
    else:
        psd = 30
    report.add_epochs(
        epochs=epochs,
        title='Epochs: before cleaning',
        psd=psd,
        drop_log_ignore=()
    )

    ###########################################################################
    #
    # Visualize effect of ICA artifact rejection.
    #
    if cfg.spatial_filter == 'ica':
        msg = 'Adding ICA to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )
        epochs = mne.read_epochs(fname_epo_not_clean)
        ica = mne.preprocessing.read_ica(fname_ica)
        ica_reject = _get_reject(
            subject=subject,
            session=session,
            reject=cfg.ica_reject,
            ch_types=cfg.ch_types,
            param='ica_reject',
        )
        # TODO: Ref is set during ICA epochs fitting, we should ensure we do
        # it here, too
        epochs.drop_bad(ica_reject)

        if ica.exclude:
            report.add_ica(
                ica=ica,
                title='ICA',
                inst=epochs,
                picks=ica.exclude
                # TODO upstream
                # captions=f'Evoked response (across all epochs) '
                # f'before and after ICA '
                # f'({len(ica.exclude)} ICs removed)'
            )

    ###########################################################################
    #
    # Visualize effect of SSP artifact rejection.
    #

    if cfg.spatial_filter == 'ssp':
        fnames = dict(ecg=fname_ecg_epochs, eog=fname_eog_epochs)
        for kind, fname in fnames.items():
            if not fname.fpath.is_file():
                continue
            msg = f'Adding {kind.upper()} SSP to report.'
            logger.info(
                **gen_log_kwargs(
                    message=msg, subject=subject, session=session
                )
            )
            # Eventually we should add this to report somehow
            epochs = mne.read_epochs(fname)
            projs = mne.read_proj(fname_ssp)
            projs = [p for p in projs if kind.upper() in p['desc']]
            assert len(projs), len(projs)  # should exist if the epochs do
            picks_trace = None
            if kind == 'ecg':
                if 'ecg' in epochs:
                    picks_trace = 'ecg'
            else:
                assert kind == 'eog'
                if cfg.eog_channels:
                    picks_trace = cfg.eog_channels
                elif 'eog' in epochs:
                    picks_trace = 'eog'
            fig = mne.viz.plot_projs_joint(
                projs, epochs.average(picks='all'), picks_trace=picks_trace)
            caption = (
                f'Computed using {len(epochs)} epochs '
                f'(from {len(epochs.drop_log)} original events)'
            )
            report.add_figure(
                fig, title=f'SSP: {kind.upper()}', caption=caption,
                tags=('ssp', kind))
            plt.close(fig)

    ###########################################################################
    #
    # Visualize cleaned epochs.
    #
    msg = 'Adding cleaned epochs to report.'
    logger.info(
        **gen_log_kwargs(
            message=msg, subject=subject, session=session
        )
    )
    epochs = mne.read_epochs(fname_epo_clean)
    # Add PSD plots for 30s of data or all epochs if we have less available
    if len(epochs) * (epochs.tmax - epochs.tmin) < 30:
        psd = True
    else:
        psd = 30
    report.add_epochs(
        epochs=epochs,
        title='Epochs: after cleaning',
        psd=psd,
        drop_log_ignore=()
    )

    return report


def run_report_sensor(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    report: mne.Report
) -> mne.Report:
    import matplotlib.pyplot as plt  # nested import to help joblib

    msg = 'Generating sensor-space analysis report …'
    logger.info(
        **gen_log_kwargs(message=msg, subject=subject, session=session)
    )

    if report is None:
        report = _gen_empty_report(
            cfg=cfg,
            subject=subject,
            session=session
        )

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )
    fname_epo_clean = bids_path.copy().update(
        processing='clean',
        suffix='epo'
    )
    fname_ave = bids_path.copy().update(suffix='ave')
    fname_decoding = bids_path.copy().update(
        processing=None,
        suffix='decoding',
        extension='.mat'
    )
    fname_tfr_pow = bids_path.copy().update(
        suffix='power+condition+tfr',
        extension='.h5'
    )
    fname_tfr_itc = bids_path.copy().update(
        suffix='itc+condition+tfr',
        extension='.h5'
    )
    fname_noise_cov = get_noise_cov_bids_path(
        cfg=cfg,
        subject=subject,
        session=session
    )

    ###########################################################################
    #
    # Visualize evoked responses.
    #
    if cfg.conditions is None:
        conditions = []
    elif isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()

    conditions.extend([contrast["name"] for contrast in cfg.all_contrasts])

    if conditions:
        evokeds = mne.read_evokeds(fname_ave)
    else:
        evokeds = []

    if evokeds:
        msg = (f'Adding {len(conditions)} evoked signals and contrasts to the '
               f'report.')
    else:
        msg = 'No evoked conditions or contrasts found.'

    logger.info(
        **gen_log_kwargs(message=msg, subject=subject, session=session)
    )

    if fname_noise_cov.fpath.exists():
        msg = f'Reading noise covariance: {fname_noise_cov.basename}'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )
        noise_cov = fname_noise_cov
    else:
        msg = 'No noise covariance matrix found, not rendering whitened data'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )
        noise_cov = None

    for condition, evoked in zip(conditions, evokeds):
        _restrict_analyze_channels(evoked, cfg)

        tags = ('evoked', _sanitize_cond_tag(condition))
        if condition in cfg.conditions:
            title = f'Condition: {condition}'
        else:  # It's a contrast of two conditions.
            title = f'Contrast: {condition}'
            tags = tags + ('contrast',)

        report.add_evokeds(
            evokeds=evoked,
            titles=title,
            noise_cov=noise_cov,
            n_time_points=cfg.report_evoked_n_time_points,
            tags=tags,
        )

    ###########################################################################
    #
    # Visualize full-epochs decoding results.
    #
    decode = cfg.decode and cfg.decoding_contrasts
    if decode:
        msg = 'Adding full-epochs decoding results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )

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
        )
        title = f'Full-epochs decoding: {cond_1} vs. {cond_2}'
        report.add_figure(
            fig=fig,
            title=title,
            caption=caption,
            section='Decoding: full-epochs',
            tags=(
                'epochs',
                'contrast',
                'decoding',
                *[f'{_sanitize_cond_tag(cond_1)}–'
                  f'{_sanitize_cond_tag(cond_2)}'
                  for cond_1, cond_2 in cfg.decoding_contrasts]
            )
        )
        # close figure to save memory
        plt.close(fig)
        del fig, caption, title

    ###########################################################################
    #
    # Visualize time-by-time decoding results.
    #
    if decode:
        msg = 'Adding time-by-time decoding results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )

        epochs = mne.read_epochs(fname_epo_clean)

        section = 'Decoding: time-by-time'
        for contrast in cfg.decoding_contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            tags = (
                'epochs',
                'contrast',
                'decoding',
                f"{_sanitize_cond_tag(contrast[0])}–"
                f"{_sanitize_cond_tag(contrast[1])}"
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

            fig = _plot_time_by_time_decoding_scores(
                times=decoding_data['times'].ravel(),
                cross_val_scores=decoding_data['scores'],
                metric=cfg.decoding_metric,
                time_generalization=cfg.decoding_time_generalization,
                decim=decoding_data['decim'].item(),
            )
            caption = (
                f'Time-by-time decoding: '
                f'{len(epochs[cond_1])} × {cond_1} vs. '
                f'{len(epochs[cond_2])} × {cond_2}'
            )
            title = f'Decoding over time: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                section=section,
                tags=tags,
            )
            plt.close(fig)

            if cfg.decoding_time_generalization:
                fig = _plot_decoding_time_generalization(
                    decoding_data=decoding_data,
                    metric=cfg.decoding_metric,
                    kind='single-subject'
                )
                caption = (
                    'Time generalization (generalization across time, GAT): '
                    'each classifier is trained on each time point, and '
                    'tested on all other time points.'
                )
                title = f'Time generalization: {cond_1} vs. {cond_2}'
                report.add_figure(
                    fig=fig,
                    title=title,
                    caption=caption,
                    section=section,
                    tags=tags,
                )
                plt.close(fig)

            del decoding_data, cond_1, cond_2, caption

        del epochs

    ###########################################################################
    #
    # Visualize CSP decoding results.
    #

    if decode and cfg.decoding_csp:
        msg = 'Adding CSP decoding results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )
        section = 'Decoding: CSP'
        all_csp_tf_results = dict()
        for contrast in cfg.decoding_contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            tags = (
                'epochs',
                'contrast',
                'decoding',
                'csp',
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}"
            )
            processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding = bids_path.copy().update(
                processing=processing,
                suffix='decoding',
                extension='.xlsx'
            )
            assert fname_decoding.fpath.is_file(), fname_decoding.fpath
            csp_freq_results = pd.read_excel(
                fname_decoding,
                sheet_name='CSP Frequency'
            )
            csp_freq_results['scores'] = csp_freq_results['scores'].apply(
                lambda x: np.array(x[1:-1].split(), float))
            csp_tf_results = pd.read_excel(
                fname_decoding,
                sheet_name='CSP Time-Frequency'
            )
            csp_tf_results['scores'] = csp_tf_results['scores'].apply(
                lambda x: np.array(x[1:-1].split(), float))
            all_csp_tf_results[contrast] = csp_tf_results
            del csp_tf_results
            all_decoding_scores = list()
            contrast_names = list()
            for freq_range_name in cfg.decoding_csp_freqs.keys():
                results = csp_freq_results.loc[
                    csp_freq_results['freq_range_name'] == freq_range_name, :
                ]
                all_decoding_scores.append(results['scores'].item())
                f_min = float(results['f_min'])
                f_max = float(results['f_max'])
                contrast_names.append(
                    f'{freq_range_name}\n'
                    f'({f_min:0.1f}-{f_max:0.1f} Hz)'
                )
            fig, caption = _plot_full_epochs_decoding_scores(
                contrast_names=contrast_names,
                scores=all_decoding_scores,
                metric=cfg.decoding_metric,
            )
            title = f'CSP decoding: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                caption=caption,
                tags=tags,
            )
            # close figure to save memory
            plt.close(fig)
            del fig, caption, title

        # Now, plot decoding scores across time-frequency bins.
        for ci, contrast in enumerate(cfg.decoding_contrasts):
            cond_1, cond_2 = contrast
            tags = (
                'epochs',
                'contrast',
                'decoding',
                'csp',
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
            )
            results = all_csp_tf_results[contrast]
            mean_crossval_scores = list()
            tmin, tmax, fmin, fmax = list(), list(), list(), list()
            for freq_range_name in cfg.decoding_csp_freqs.keys():
                mean_crossval_scores.extend(
                    results['mean_crossval_score'].ravel())
                tmin.extend(results['t_min'].ravel())
                tmax.extend(results['t_max'].ravel())
                fmin.extend(results['f_min'].ravel())
                fmax.extend(results['f_max'].ravel())
            mean_crossval_scores = np.array(mean_crossval_scores, float)
            fig, ax = plt.subplots(constrained_layout=True)
            # XXX Add support for more metrics
            assert cfg.decoding_metric == 'roc_auc'
            vmax = max(
                np.abs(mean_crossval_scores.min() - 0.5),
                np.abs(mean_crossval_scores.max() - 0.5)
            ) + 0.5
            vmin = 0.5 - (vmax - 0.5)
            img = _imshow_tf(
                mean_crossval_scores, ax,
                tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                vmin=vmin, vmax=vmax)
            ax.set_xlim([np.min(tmin), np.max(tmax)])
            ax.set_ylim([np.min(fmin), np.max(fmax)])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            cbar = fig.colorbar(
                ax=ax, shrink=0.75, orientation='vertical', mappable=img)
            metric = dict(
                roc_auc='ROC AUC'
            ).get(cfg.decoding_metric, cfg.decoding_metric)
            cbar.set_label(f'Mean decoding score ({metric})')
            title = f'CSP TF decoding: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                tags=tags,
            )

    ###########################################################################
    #
    # Visualize TFR as topography.
    #
    if cfg.time_frequency_conditions is None:
        conditions = []
    elif isinstance(cfg.time_frequency_conditions, dict):
        conditions = list(cfg.time_frequency_conditions.keys())
    else:
        conditions = cfg.time_frequency_conditions.copy()

    if conditions:
        msg = 'Adding TFR analysis results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )

    for condition in conditions:
        cond = sanitize_cond_name(condition)
        fname_tfr_pow_cond = str(fname_tfr_pow.copy()).replace("+condition+",
                                                               f"+{cond}+")
        fname_tfr_itc_cond = str(fname_tfr_itc.copy()).replace("+condition+",
                                                               f"+{cond}+")
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
            tags=('time-frequency', _sanitize_cond_tag(condition))
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
            tags=('time-frequency', _sanitize_cond_tag(condition))
        )
        plt.close(fig_power)
        del itc

    return report


def run_report_source(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    report: mne.Report
) -> mne.Report:
    import matplotlib.pyplot as plt  # nested import to help joblib

    msg = 'Generating source-space analysis report …'
    logger.info(
        **gen_log_kwargs(message=msg, subject=subject, session=session)
    )

    if report is None:
        report = _gen_empty_report(
            cfg=cfg,
            subject=subject,
            session=session
        )

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    # Use this as a source for the Info dictionary
    fname_info = bids_path.copy().update(
        processing='clean',
        suffix='epo'
    )

    fname_trans = bids_path.copy().update(suffix='trans')
    if not fname_trans.fpath.exists():
        msg = 'No coregistration found, skipping source space report.'
        logger.info(**gen_log_kwargs(message=msg,
                                     subject=subject, session=session))
        return report

    fname_noise_cov = get_noise_cov_bids_path(
        cfg=cfg,
        subject=subject,
        session=session
    )

    ###########################################################################
    #
    # Visualize coregistration, noise covariance matrix, & inverse solutions.
    #

    if cfg.conditions is None:
        conditions = []
    elif isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()

    conditions.extend([contrast["name"] for contrast in cfg.all_contrasts])

    msg = 'Rendering MRI slices with BEM contours.'
    logger.info(**gen_log_kwargs(message=msg,
                                 subject=subject, session=session))
    report.add_bem(
        subject=cfg.fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        title='BEM',
        width=256,
        decim=8
    )

    msg = 'Rendering sensor alignment (coregistration).'
    logger.info(**gen_log_kwargs(message=msg,
                                 subject=subject, session=session))
    report.add_trans(
        trans=fname_trans,
        info=fname_info,
        title='Sensor alignment',
        subject=cfg.fs_subject,
        subjects_dir=cfg.fs_subjects_dir,
        alpha=1
    )

    msg = 'Rendering noise covariance matrix and corresponding SVD.'
    logger.info(**gen_log_kwargs(message=msg,
                                 subject=subject, session=session))
    report.add_covariance(
        cov=fname_noise_cov,
        info=fname_info,
        title='Noise covariance'
    )

    for condition in conditions:
        msg = f'Rendering inverse solution for {condition}'
        logger.info(**gen_log_kwargs(message=msg,
                                     subject=subject, session=session))

        if condition in cfg.conditions:
            title = f'Source: {condition}'
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        method = cfg.inverse_method
        cond_str = sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.

        fname_stc = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{hemi_str}',
            extension=None)

        tags = (
            'source-estimate',
            _sanitize_cond_tag(condition)
        )
        if Path(f'{fname_stc.fpath}-lh.stc').exists():
            report.add_stc(
                stc=fname_stc,
                title=title,
                subject=cfg.fs_subject,
                subjects_dir=cfg.fs_subjects_dir,
                n_time_points=cfg.report_stc_n_time_points,
                tags=tags
            )

    plt.close('all')  # close all figures to save memory
    return report


@failsafe_run(script_path=__file__)
def run_report(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
):
    report = _gen_empty_report(
        cfg=cfg,
        subject=subject,
        session=session
    )
    kwargs = dict(cfg=cfg, subject=subject, session=session, report=report)
    report = run_report_preprocessing(**kwargs)
    report = run_report_sensor(**kwargs)
    report = run_report_source(**kwargs)

    ###########################################################################
    #
    # Add configuration and system info.
    #
    add_system_info(report)

    ###########################################################################
    #
    # Save the report.
    #
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )
    fname_report = bids_path.copy().update(suffix='report', extension='.html')
    report.save(
        fname=fname_report,
        open_browser=cfg.interactive,
        overwrite=True
    )


def add_event_counts(*,
                     cfg,
                     session: Optional[str],
                     report: mne.Report) -> None:
    try:
        df_events = count_events(BIDSPath(root=cfg.bids_root,
                                          session=session))
    except ValueError:
        logger.warning('Could not read events.')
        df_events = None

    if df_events is not None:
        css_classes = ('table', 'table-striped', 'table-borderless',
                       'table-hover')
        report.add_html(
            f'<div class="event-counts">\n'
            f'{df_events.to_html(classes=css_classes, border=0)}\n'
            f'</div>',
            title='Event counts',
            tags=('events',)
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
        report.add_custom_css(css=css)


def add_system_info(report: mne.Report):
    """Add system information and the pipeline configuration to the report."""

    config_path = Path(os.environ['MNE_BIDS_STUDY_CONFIG'])
    report.add_code(
        code=config_path,
        title='Configuration file',
        tags=('configuration',)
    )
    report.add_sys_info(title='System information')


@failsafe_run(script_path=__file__)
def run_report_average(*, cfg, subject: str, session: str) -> None:
    # Group report
    import matplotlib.pyplot as plt  # nested import to help joblib

    msg = 'Generating grand average report …'
    logger.info(
        **gen_log_kwargs(message=msg, subject=subject, session=session)
    )

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

    report = mne.Report(
        title=title,
        raw_psd=True
    )
    evokeds = mne.read_evokeds(evoked_fname)
    for evoked in evokeds:
        _restrict_analyze_channels(evoked, cfg)

    method = cfg.inverse_method
    inverse_str = method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph2fsaverage'

    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()

    conditions.extend([contrast["name"] for contrast in cfg.all_contrasts])

    #######################################################################
    #
    # Add event stats.
    #
    add_event_counts(cfg=cfg, report=report, session=session)

    #######################################################################
    #
    # Visualize evoked responses.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in cfg.conditions:
            title = f'Average: {condition}'
            tags = (
                'evoked',
                _sanitize_cond_tag(condition)
            )
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        report.add_evokeds(
            evokeds=evoked,
            titles=title,
            projs=False,
            tags=tags,
            n_time_points=cfg.report_evoked_n_time_points,
            # captions=evoked.comment  # TODO upstream
        )

    #######################################################################
    #
    # Visualize decoding results.
    #
    if cfg.decode and cfg.decoding_contrasts:
        add_decoding_grand_average(
            session=session, cfg=cfg, report=report
        )

    if cfg.decode and cfg.decoding_csp:
        add_csp_grand_average(
            session=session, cfg=cfg, report=report
        )

    #######################################################################
    #
    # Visualize forward solution, inverse operator, and inverse solutions.
    #

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

        if Path(f'{fname_stc_avg.fpath}-lh.stc').exists():
            report.add_stc(
                stc=fname_stc_avg,
                title=title,
                subject='fsaverage',
                subjects_dir=cfg.fs_subjects_dir,
                n_time_points=cfg.report_stc_n_time_points,
                tags=tags
            )

    ###########################################################################
    #
    # Add configuration and system info.
    #
    add_system_info(report)

    ###########################################################################
    #
    # Save the report.
    #
    fname_report = evoked_fname.copy().update(
        task=cfg.task, suffix='report', extension='.html')
    report.save(fname=fname_report, open_browser=False, overwrite=True)

    plt.close('all')


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
        )
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
        freq_names = list(cfg.decoding_csp_freqs)
        for freq_range_name in cfg.decoding_csp_freqs:
            results = csp_freq_results.loc[
                csp_freq_results['freq_range_name'] == freq_range_name, :
            ]
            freq_bin_starts.append(results['f_min'].item())
            freq_bin_widths.append(
                (results['f_max'] - results['f_min']).item())
            decoding_scores.append(results['mean'].item())
            cis_lower = results['mean_ci_lower'].item()
            cis_upper = results['mean_ci_upper'].item()
            error_bars_lower = decoding_scores[-1] - cis_lower
            error_bars_upper = cis_upper - decoding_scores[-1]
            error_bars.append(np.stack([error_bars_lower, error_bars_upper]))
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
        for name, start, width in zip(
                freq_names, freq_bin_starts, freq_bin_widths):
            ax.text(
                x=start + width / 2,
                y=0.02,
                s=name,
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
        for freq_range_name in cfg.decoding_csp_freqs.keys():
            results = csp_cluster_results[freq_range_name][0][0]
            mean_crossval_scores = results['mean_crossval_scores'].ravel()
            # t_vals = results['t_vals']
            clusters = results['clusters']
            cluster_p_vals = results['cluster_p_vals'].squeeze()
            tmin = results['time_bin_edges'].ravel()
            tmin, tmax = tmin[:-1], tmin[1:]
            lims[0] = min(lims[0], tmin.min())
            lims[1] = max(lims[1], tmax.max())
            fmin, fmax = results['freq_bin_edges'].ravel()
            lims[2] = min(lims[2], fmin.min())
            lims[3] = max(lims[3], fmax.max())
            fmin = np.repeat(fmin, len(mean_crossval_scores))
            fmax = np.repeat(fmax, len(mean_crossval_scores))
            cluster_t_threshold = results['cluster_t_threshold'].ravel().item()

            significant_cluster_idx = np.where(
                cluster_p_vals < cfg.cluster_permutation_p_threshold
            )[0]
            significant_clusters = clusters[significant_cluster_idx]
            n_clu += len(significant_cluster_idx)

            # XXX Add support for more metrics
            assert cfg.decoding_metric == 'roc_auc'
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
                if cfg.decoding_metric == 'roc_auc':
                    metric = 'ROC AUC'
                cbar.set_label(f'Mean decoding score ({metric})')
            ax[0].text(0.9 * tmin[0] + 0.1 * tmax[0],
                       0.5 * fmin[0] + 0.5 * fmax[0],
                       freq_range_name,
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
                    f'significant cluster(s) '
                    f'(p < {cfg.cluster_permutation_p_threshold}; '
                    f'bins with absolute t-values > '
                    f'{round(cluster_t_threshold, 3)} '
                    f'were used to form clusters).',
            tags=tags,
        )


def get_config(
    *,
    config,
    subject: str,
) -> SimpleNamespace:
    # Deal with configurations where `deriv_root` was specified, but not
    # `fs_subjects_dir`. We normally raise an exception in this case in
    # `get_fs_subjects_dir()`. However, in situations where users only run the
    # sensor-space scripts, we never call this function, so everything works
    # totally fine at first (which is expected). Yet, when creating the
    # reports, the pipeline would fail with an exception – which is
    # unjustified, as it would not make sense to force users to provide an
    # `fs_subjects_dir` if they don't care about source analysis anyway! So
    # simply assign a dummy value in such cases.
    # `get_fs_subject()` calls `get_fs_subjects_dir()`, so take care of this
    # too.
    try:
        fs_subjects_dir = get_fs_subjects_dir(config)
    except ValueError:
        fs_subjects_dir = None
        fs_subject = None
    else:
        fs_subject = get_fs_subject(config=config, subject=subject)

    dtg_decim = config.decoding_time_generalization_decim
    cfg = SimpleNamespace(
        task=get_task(config),
        task_is_rest=config.task_is_rest,
        runs=get_runs(config=config, subject=subject),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        analyze_channels=config.analyze_channels,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        h_freq=config.h_freq,
        spatial_filter=config.spatial_filter,
        conditions=config.conditions,
        all_contrasts=get_all_contrasts(config),
        decoding_contrasts=get_decoding_contrasts(config),
        ica_reject=config.ica_reject,
        ch_types=config.ch_types,
        time_frequency_conditions=config.time_frequency_conditions,
        time_frequency_baseline=config.time_frequency_baseline,
        time_frequency_baseline_mode=config.time_frequency_baseline_mode,
        time_frequency_crop=config.time_frequency_crop,
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_time_generalization=config.decoding_time_generalization,
        decoding_time_generalization_decim=dtg_decim,
        decoding_csp=config.decoding_csp,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        n_boot=config.n_boot,
        cluster_permutation_p_threshold=config.cluster_permutation_p_threshold,
        cluster_forming_t_threshold=config.cluster_forming_t_threshold,
        inverse_method=config.inverse_method,
        report_stc_n_time_points=config.report_stc_n_time_points,
        report_evoked_n_time_points=config.report_evoked_n_time_points,
        fs_subject=fs_subject,
        fs_subjects_dir=fs_subjects_dir,
        deriv_root=get_deriv_root(config),
        bids_root=get_bids_root(config),
        use_template_mri=config.use_template_mri,
        interactive=config.interactive,
        plot_psd_for_runs=config.plot_psd_for_runs,
        eog_channels=config.eog_channels,
        noise_cov=_sanitize_callable(config.noise_cov),
        data_type=config.data_type,
        subjects=config.subjects,
        exclude_subjects=config.exclude_subjects,
    )
    return cfg


@contextlib.contextmanager
def _agg_backend():
    import matplotlib
    backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)
    try:
        yield
    finally:
        matplotlib.use(backend, force=True)


def main():
    """Make reports."""
    import config
    with get_parallel_backend(config), _agg_backend():
        parallel, run_func = parallel_func(run_report, config=config)
        sessions = get_sessions(config=config)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                subject=subject,
                session=session
            )
            for subject in get_subjects(config=config)
            for session in sessions
        )

        if config.task_is_rest:
            msg = '    … skipping "average" report for "rest" task.'
            logger.info(**gen_log_kwargs(message=msg))
            avg_subjects = []
        else:
            avg_subjects = ['average']

        parallel, run_func = parallel_func(run_report_average, config=config)
        logs.extend(parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                subject=subject,
                session=session,
            )
            for subject in avg_subjects
            for session in sessions
        ))
        save_logs(logs=logs, config=config)


if __name__ == '__main__':
    main()
