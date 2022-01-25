"""
================
99. Make reports
================

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

import os.path as op
from pathlib import Path
import itertools
import logging
from typing import Tuple, Union, Optional
from types import SimpleNamespace

from scipy.io import loadmat
import mne
from mne_bids import BIDSPath
from mne_bids.stats import count_events

import config
from config import gen_log_kwargs, on_error, failsafe_run
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')

Condition_T = Union[str, Tuple[str]]


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

        if this_raw_fname.copy().update(split='01').fpath.exists():
            this_raw_fname.update(split='01')

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

    if raw_fname.copy().update(split='01').fpath.exists():
        raw_fname.update(split='01')

    return raw_fname


def plot_auto_scores(cfg, subject, session):
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

        figs = config.plot_auto_scores(auto_scores)
        all_figs.extend(figs)

        # Could be more than 1 fig, e.g. "grad" and "mag"
        captions = [f'Run {run}'] * len(figs)
        all_captions.extend(captions)

    return all_figs, all_captions


def plot_decoding_scores(times, cross_val_scores, metric):
    """Plot cross-validation results from time-by-time decoding.
    """
    import matplotlib.pyplot as plt

    mean_scores = cross_val_scores.mean(axis=0)
    max_scores = cross_val_scores.max(axis=0)
    min_scores = cross_val_scores.min(axis=0)

    fig, ax = plt.subplots()
    ax.axhline(0.5, ls='--', lw=0.5, color='black', label='chance')
    if times.min() < 0 < times.max():
        ax.axvline(0, ls='-', lw=0.5, color='black')
    ax.fill_between(x=times, y1=min_scores, y2=max_scores, color='lightgray',
                    alpha=0.5, label='range [min, max]')
    ax.plot(times, mean_scores, ls='-', lw=2, color='black',
            label='mean')

    ax.set_xlabel('Time (s)')
    if metric == 'roc_auc':
        metric = 'ROC AUC'
    ax.set_ylabel(f'Score ({metric})')
    ax.set_ylim((-0.025, 1.025))
    ax.legend(loc='lower right')
    fig.tight_layout()

    return fig


def plot_decoding_scores_gavg(cfg, decoding_data):
    """Plot the grand-averaged decoding scores.
    """
    import matplotlib.pyplot as plt

    # We squeeze() to make Matplotlib happy.
    times = decoding_data['times'].squeeze()
    mean_scores = decoding_data['mean'].squeeze()
    se_lower = mean_scores - decoding_data['mean_se'].squeeze()
    se_upper = mean_scores + decoding_data['mean_se'].squeeze()
    ci_lower = decoding_data['mean_ci_lower'].squeeze()
    ci_upper = decoding_data['mean_ci_upper'].squeeze()
    metric = cfg.decoding_metric

    fig, ax = plt.subplots()
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

    ax.set_xlabel('Time (s)')
    if metric == 'roc_auc':
        metric = 'ROC AUC'
    ax.set_ylabel(f'Score ({metric})')
    ax.set_ylim((-0.025, 1.025))
    ax.legend(loc='lower right')
    fig.tight_layout()

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


def run_report_preprocessing(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str] = None,
    report: Optional[mne.Report]
) -> mne.Report:
    import matplotlib.pyplot as plt

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
        if fname.copy().update(split='01').fpath.exists():
            fname.update(split='01')

        fnames_raw_filt.append(fname)

    fname_epo_not_clean = bids_path.copy().update(suffix='epo')
    fname_epo_clean = bids_path.copy().update(processing='clean', suffix='epo')
    fname_ica = bids_path.copy().update(suffix='ica')

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
            tags=('raw', 'filtered', f'run-{fname.run}')
            # caption=fname.fpath.name  # TODO upstream
        )
        del plot_raw_psd

    if cfg.process_er:
        msg = 'Adding filtered empty-room raw data to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )

        er_path = get_er_path(cfg=cfg, subject=subject, session=session)
        report.add_raw(
            raw=er_path,
            title='Empty-Room',
            butterfly=5,
            tags=('raw', 'empty-room')
            # caption=er_path.fpath.name  # TODO upstream
        )

    # Visualize automated noisy channel detection.
    if cfg.find_noisy_channels_meg:
        msg = 'Adding visualization of noisy channel detection to report.'
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=subject, session=session
            )
        )

        figs, captions = plot_auto_scores(cfg=cfg, subject=subject,
                                          session=session)

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
    if cfg.task.lower() != 'rest':
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
        title='Epochs (before cleaning)',
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
        epochs.drop_bad(cfg.ica_reject)
        ica = mne.preprocessing.read_ica(fname_ica)

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
        title='Epochs (after cleaning)',
        psd=psd,
        drop_log_ignore=()
    )

    return report


def run_report_sensor(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str] = None,
    report: mne.Report
) -> mne.Report:
    import matplotlib.pyplot as plt

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

    conditions.extend(cfg.contrasts)

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

    for condition, evoked in zip(conditions, evokeds):
        if cfg.analyze_channels:
            evoked.pick(cfg.analyze_channels)

        if condition in cfg.conditions:
            title = f'Condition: {condition}'
            tags = ('evoked', condition.lower().replace(' ', '-'))
        else:  # It's a contrast of two conditions.
            title = f'Contrast: {condition[0]} – {condition[1]}'
            tags = (
                'evoked',
                'contrast',
                f"{condition[0].lower().replace(' ', '-')}-"
                f"{condition[1].lower().replace(' ', '-')}"
            )

        report.add_evokeds(
            evokeds=evoked,
            titles=title,
            tags=tags
        )

    ###########################################################################
    #
    # Visualize decoding results.
    #
    if cfg.decode:
        msg = 'Adding time-by-time decoding results to the report.'
        logger.info(
            **gen_log_kwargs(message=msg, subject=subject, session=session)
        )

        epochs = mne.read_epochs(fname_epo_clean)

        for contrast in cfg.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            processing = f'{a_vs_b}+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding_ = (fname_decoding.copy()
                               .update(processing=processing))
            decoding_data = loadmat(fname_decoding_)
            del fname_decoding_, processing, a_vs_b

            fig = plot_decoding_scores(
                times=epochs.times,
                cross_val_scores=decoding_data['scores'],
                metric=cfg.decoding_metric
            )

            title = f'Time-by-time Decoding: {cond_1} ./. {cond_2}'
            caption = (f'{len(epochs[cond_1])} × {cond_1} ./. '
                       f'{len(epochs[cond_2])} × {cond_2}')
            tags = (
                'epochs',
                'contrast',
                f"{contrast[0].lower().replace(' ', '-')}-"
                f"{contrast[1].lower().replace(' ', '-')}"
            )

            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                tags=tags
            )
            plt.close(fig)
            del decoding_data, cond_1, cond_2, title, caption

        del epochs

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
        cond = config.sanitize_cond_name(condition)
        fname_tfr_pow_cond = str(fname_tfr_pow.copy()).replace("+condition+",
                                                               f"+{cond}+")
        power = mne.time_frequency.read_tfrs(fname_tfr_pow_cond)
        fig = power[0].plot_topo(show=False, fig_facecolor='w', font_color='k',
                                 border='k')
        report.add_figure(
            fig=fig,
            title=f'TFR: {condition}',
            caption=f'TFR Power: {condition}',
            tags=('time-frequency', condition.lower().replace(' ', '-'))
        )
        plt.close(fig)

    return report


def run_report_source(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str] = None,
    report: mne.Report
) -> mne.Report:
    import matplotlib.pyplot as plt

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

    ###########################################################################
    #
    # Visualize the coregistration & inverse solutions.
    #

    if cfg.conditions is None:
        conditions = []
    elif isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()

    conditions.extend(cfg.contrasts)

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
    )

    for condition in conditions:
        msg = f'Rendering inverse solution for {condition}'
        logger.info(**gen_log_kwargs(message=msg,
                                     subject=subject, session=session))

        if condition in cfg.conditions:
            title = f'Source: {config.sanitize_cond_name(condition)}'
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        method = cfg.inverse_method
        cond_str = config.sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.

        fname_stc = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{hemi_str}',
            extension=None)

        tags = (
            'source-estimate',
            condition.lower().replace(' ', '-')
        )
        if Path(f'{fname_stc.fpath}-lh.stc').exists():
            report.add_stc(
                stc=fname_stc,
                title=title,
                subject=cfg.fs_subject,
                subjects_dir=cfg.fs_subjects_dir,
                tags=tags
            )

    plt.close('all')  # close all figures to save memory
    return report


@failsafe_run(on_error=on_error, script_path=__file__)
def run_report(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str] = None,
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
                     session: str,
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


@failsafe_run(on_error=on_error, script_path=__file__)
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
    if cfg.analyze_channels:
        for evoked in evokeds:
            evoked.pick(cfg.analyze_channels)

    method = cfg.inverse_method
    inverse_str = method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph2fsaverage'

    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions.copy()

    conditions.extend(cfg.contrasts)

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
                config.sanitize_cond_name(condition).lower().replace(' ', '')
            )
        else:  # It's a contrast of two conditions.
            title = f'Average Contrast: {condition[0]} – {condition[1]}'
            tags = (
                'evoked',
                f'{config.sanitize_cond_name(condition[0])} – '
                f'{config.sanitize_cond_name(condition[1])}'
                .lower().replace(' ', '')
            )

        report.add_evokeds(
            evokeds=evoked,
            titles=title,
            projs=False,
            tags=tags,
            # captions=evoked.comment  # TODO upstream
        )

    #######################################################################
    #
    # Visualize decoding results.
    #
    if cfg.decode:
        for contrast in cfg.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            processing = f'{a_vs_b}+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding_ = evoked_fname.copy().update(
                processing=processing,
                suffix='decoding',
                extension='.mat'
            )
            decoding_data = loadmat(fname_decoding_)
            del fname_decoding_, processing, a_vs_b

            fig = plot_decoding_scores_gavg(cfg=cfg,
                                            decoding_data=decoding_data)
            title = f'Time-by-time Decoding: {cond_1} ./. {cond_2}'
            caption = (f'Based on N={decoding_data["N"].squeeze()} '
                       f'subjects. Standard error and confidence interval '
                       f'of the mean were bootstrapped with {cfg.n_boot} '
                       f'resamples.')
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                tags=(
                    'decoding',
                    'contrast',
                    f'{config.sanitize_cond_name(cond_1)} – '
                    f'{config.sanitize_cond_name(cond_2)}'
                    .lower().replace(' ', '-')
                )
            )
            plt.close(fig)
            del decoding_data, cond_1, cond_2, caption, title

    #######################################################################
    #
    # Visualize forward solution, inverse operator, and inverse solutions.
    #

    for condition, evoked in zip(conditions, evokeds):
        if condition in cfg.conditions:
            title = f'Average: {condition}'
            cond_str = config.sanitize_cond_name(condition)
            tags = (
                'source-estimate',
                config.sanitize_cond_name(condition).lower().replace(' ', '')
            )
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        fname_stc_avg = evoked_fname.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}',
            extension=None)

        if Path(f'{fname_stc_avg.fpath}-lh.stc').exists():
            report.add_stc(
                stc=fname_stc_avg,
                title=title,
                subject='fsaverage',
                subjects_dir=cfg.fs_subjects_dir,
                tags=tags
            )

    fname_report = evoked_fname.copy().update(
        task=cfg.task, suffix='report', extension='.html')
    report.save(fname=fname_report, open_browser=False, overwrite=True)

    plt.close('all')  # close all figures to save memory


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
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
        fs_subjects_dir = config.get_fs_subjects_dir()
    except ValueError:
        fs_subjects_dir = None
        fs_subject = None
    else:
        fs_subject = config.get_fs_subject(subject=subject)

    cfg = SimpleNamespace(
        task=config.get_task(),
        runs=config.get_runs(subject=subject),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        analyze_channels=config.analyze_channels,
        process_er=config.process_er,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        h_freq=config.h_freq,
        spatial_filter=config.spatial_filter,
        conditions=config.conditions,
        contrasts=config.contrasts,
        ica_reject=config.get_ica_reject(),
        time_frequency_conditions=config.time_frequency_conditions,
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        n_boot=config.n_boot,
        inverse_method=config.inverse_method,
        fs_subject=fs_subject,
        fs_subjects_dir=fs_subjects_dir,
        deriv_root=config.get_deriv_root(),
        bids_root=config.get_bids_root(),
        use_template_mri=config.use_template_mri,
        interactive=config.interactive,
        plot_psd_for_runs=config.plot_psd_for_runs,
    )
    return cfg


def main():
    """Make reports."""
    with config.get_parallel_backend():
        parallel, run_func, _ = parallel_func(
            run_report,
            n_jobs=config.get_n_jobs()
        )
        logs = parallel(
            run_func(
                cfg=get_config(subject=subject), subject=subject,
                session=session
            )
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )

        config.save_logs(logs)

        sessions = config.get_sessions()
        if not sessions:
            sessions = [None]

        if (config.get_task() is not None and
                config.get_task().lower() == 'rest'):
            msg = '    … skipping "average" report for "rest" task.'
            logger.info(**gen_log_kwargs(message=msg))
            return

        for session in sessions:
            run_report_average(
                cfg=get_config(subject='average'),
                subject='average',
                session=session
            )


if __name__ == '__main__':
    main()
