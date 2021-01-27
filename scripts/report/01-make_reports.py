"""
================
99. Make reports
================

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

import os.path as op
import itertools
import logging
from typing import Dict, Any, List, Tuple, Union

import numpy as np
from scipy.io import loadmat
import matplotlib

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath
from mne_bids.stats import count_events

import config
from config import gen_log_message, on_error, failsafe_run

matplotlib.use('Agg')  # do not open any window  # noqa

logger = logging.getLogger('mne-study-template')

Condition_T = Union[str, Tuple[str]]


def plot_events(subject, session):
    raws_filt = []
    raw_fname = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         recording=config.rec,
                         space=config.space,
                         processing='filt',
                         suffix='raw',
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    for run in config.get_runs():
        this_raw_fname = raw_fname.copy().update(run=run)

        if this_raw_fname.copy().update(split='01').fpath.exists():
            this_raw_fname.update(split='01')

        raw_filt = mne.io.read_raw_fif(this_raw_fname)
        raws_filt.append(raw_filt)
        del this_raw_fname

    # Concatenate the filtered raws and extract the events.
    raw_filt_concat = mne.concatenate_raws(raws_filt)
    events, event_id = mne.events_from_annotations(raw=raw_filt_concat)
    fig = mne.viz.plot_events(events=events, event_id=event_id,
                              first_samp=raw_filt_concat.first_samp,
                              sfreq=raw_filt_concat.info['sfreq'],
                              show=False)
    return fig


def plot_er_psd(subject, session):
    raw_fname = BIDSPath(subject=subject,
                         session=session,
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         task='noise',
                         processing='filt',
                         suffix='raw',
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    extra_params = dict()
    if not config.use_maxwell_filter and config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    if raw_fname.copy().update(split='01').fpath.exists():
        raw_fname.update(split='01')

    raw_er_filtered = mne.io.read_raw_fif(raw_fname, preload=True,
                                          **extra_params)

    fmax = 1.5 * config.h_freq if config.h_freq is not None else np.inf
    fig = raw_er_filtered.plot_psd(fmax=fmax, show=False)
    return fig


def plot_auto_scores(subject, session):
    """Plot automated bad channel detection scores.
    """
    import json_tricks

    fname_scores = BIDSPath(subject=subject,
                            session=session,
                            task=config.get_task(),
                            acquisition=config.acq,
                            run=None,
                            processing=config.proc,
                            recording=config.rec,
                            space=config.space,
                            suffix='scores',
                            extension='.json',
                            datatype=config.get_datatype(),
                            root=config.deriv_root,
                            check=False)

    all_figs = []
    all_captions = []
    for run in config.get_runs():
        with open(fname_scores.update(run=run), 'r') as f:
            auto_scores = json_tricks.load(f)

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


def plot_decoding_scores_gavg(decoding_data):
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
    metric = config.decoding_metric

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


def run_report(subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    fname_ave = bids_path.copy().update(suffix='ave')
    fname_trans = bids_path.copy().update(suffix='trans')
    fname_epo = bids_path.copy().update(suffix='epo')
    fname_trans = bids_path.copy().update(suffix='trans')
    fname_ica = bids_path.copy().update(suffix='ica')
    fname_decoding = fname_epo.copy().update(suffix='decoding',
                                             extension='.mat')

    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()

    params: Dict[str, Any] = dict(info_fname=fname_ave, raw_psd=True)
    if op.exists(fname_trans):
        params['subject'] = fs_subject
        params['subjects_dir'] = fs_subjects_dir

    rep = mne.Report(**params)
    rep_kwargs: Dict[str, Any] = dict(data_path=fname_ave.fpath.parent,
                                      verbose=False)
    if not op.exists(fname_trans):
        rep_kwargs['render_bem'] = False

    task = config.get_task()
    if task is not None:
        rep_kwargs['pattern'] = f'*_task-{task}*'
    if mne.viz.get_3d_backend() is not None:
        with mne.viz.use_3d_backend('pyvista'):
            rep.parse_folder(**rep_kwargs)
    else:
        rep.parse_folder(**rep_kwargs)

    # Visualize automated noisy channel detection.
    if config.find_noisy_channels_meg:
        figs, captions = plot_auto_scores(subject=subject, session=session)
        rep.add_figs_to_section(figs=figs,
                                captions=captions,
                                section='Data Quality')

    # Visualize events.
    events_fig = plot_events(subject=subject, session=session)
    rep.add_figs_to_section(figs=events_fig,
                            captions='Events in filtered continuous data',
                            section='Events')

    ###########################################################################
    #
    # Visualize effect of ICA artifact rejection.
    #
    if config.use_ica:
        epochs = mne.read_epochs(fname_epo)
        ica = mne.preprocessing.read_ica(fname_ica)
        fig = ica.plot_overlay(epochs.average(), show=False)
        rep.add_figs_to_section(fig,
                                captions='Evoked response (across all epochs) '
                                         'before and after ICA',
                                section='ICA')

    ###########################################################################
    #
    # Visualize evoked responses.
    #
    conditions: List[Condition_T] = list(config.conditions)
    conditions.extend(config.contrasts)
    evokeds = mne.read_evokeds(fname_ave)
    if config.analyze_channels:
        for evoked in evokeds:
            evoked.pick(config.analyze_channels)

    for condition, evoked in zip(conditions, evokeds):
        if condition in config.conditions:
            caption = f'Condition: {condition}'
            section = 'Evoked'
        else:  # It's a contrast of two conditions.
            caption = f'Contrast: {condition[0]} – {condition[1]}'
            section = 'Contrast'

        fig = evoked.plot(spatial_colors=True, gfp=True, show=False)
        rep.add_figs_to_section(figs=fig, captions=caption,
                                comments=evoked.comment, section=section)

    ###########################################################################
    #
    # Visualize decoding results.
    #
    if config.decode:
        epochs = mne.read_epochs(fname_epo)

        for contrast in config.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}-{cond_2}'.replace(op.sep, '')
            processing = f'{a_vs_b}+{config.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding_ = (fname_decoding.copy()
                               .update(processing=processing))
            decoding_data = loadmat(fname_decoding_)
            del fname_decoding_, processing, a_vs_b

            fig = plot_decoding_scores(
                times=epochs.times,
                cross_val_scores=decoding_data['scores'],
                metric=config.decoding_metric)

            caption = f'Time-by-time Decoding: {cond_1} ./. {cond_2}'
            comment = (f'{len(epochs[cond_1])} × {cond_1} ./. '
                       f'{len(epochs[cond_2])} × {cond_2}')
            rep.add_figs_to_section(figs=fig, captions=caption,
                                    comments=comment,
                                    section='Decoding')
            del decoding_data, cond_1, cond_2, caption, comment

        del epochs

    ###########################################################################
    #
    # Visualize the coregistration & inverse solutions.
    #
    evokeds = mne.read_evokeds(fname_ave)

    if op.exists(fname_trans):
        # We can only plot the coregistration if we have a valid 3d backend.
        if mne.viz.get_3d_backend() is not None:
            fig = mne.viz.plot_alignment(evoked.info, fname_trans,
                                         subject=fs_subject,
                                         subjects_dir=fs_subjects_dir,
                                         meg=True, dig=True, eeg=True)
            rep.add_figs_to_section(figs=fig, captions='Coregistration',
                                    section='Coregistration')
        else:
            msg = ('Cannot render sensor alignment (coregistration) because '
                   'no usable 3d backend was found.')
            logger.warning(gen_log_message(message=msg, step=99,
                                           subject=subject, session=session))

        for condition, evoked in zip(conditions, evokeds):
            msg = f'Rendering inverse solution for {evoked.comment} …'
            logger.info(gen_log_message(message=msg, step=99,
                                        subject=subject, session=session))

            if condition in config.conditions:
                full_condition = config.sanitize_cond_name(evoked.comment)
                caption = f'Condition: {full_condition}'
                del full_condition
            else:  # It's a contrast of two conditions.
                # XXX Will change once we process contrasts here too
                continue

            method = config.inverse_method
            cond_str = config.sanitize_cond_name(condition)
            inverse_str = method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.

            fname_stc = bids_path.copy().update(
                suffix=f'{cond_str}+{inverse_str}+{hemi_str}',
                extension=None)

            if op.exists(str(fname_stc) + "-lh.stc"):
                stc = mne.read_source_estimate(fname_stc,
                                               subject=fs_subject)
                _, peak_time = stc.get_peak()

                # Plot using 3d backend if available, and use Matplotlib
                # otherwise.
                import matplotlib.pyplot as plt

                if mne.viz.get_3d_backend() is not None:
                    brain = stc.plot(views=['lat'], hemi='split',
                                     initial_time=peak_time, backend='pyvista',
                                     time_viewer=True,
                                     subjects_dir=fs_subjects_dir)
                    brain.toggle_interface()
                    brain._renderer.plotter.reset_camera()
                    brain._renderer.plotter.subplot(0, 0)
                    brain._renderer.plotter.reset_camera()
                    figs, ax = plt.subplots(figsize=(15, 10))
                    ax.imshow(brain.screenshot(time_viewer=True))
                    ax.axis('off')
                    comments = evoked.comment
                    captions = caption
                else:
                    fig_lh = plt.figure()
                    fig_rh = plt.figure()

                    brain_lh = stc.plot(views='lat', hemi='lh',
                                        initial_time=peak_time,
                                        backend='matplotlib',
                                        subjects_dir=fs_subjects_dir,
                                        figure=fig_lh)
                    brain_rh = stc.plot(views='lat', hemi='rh',
                                        initial_time=peak_time,
                                        subjects_dir=fs_subjects_dir,
                                        backend='matplotlib',
                                        figure=fig_rh)
                    figs = [brain_lh, brain_rh]
                    comments = [f'{evoked.comment} - left hemisphere',
                                f'{evoked.comment} - right hemisphere']
                    captions = [f'{caption} - left',
                                f'{caption} - right']

                rep.add_figs_to_section(figs=figs,
                                        captions=captions,
                                        comments=comments,
                                        section='Sources')
                del peak_time

    if config.process_er:
        fig_er_psd = plot_er_psd(subject=subject, session=session)
        rep.add_figs_to_section(figs=fig_er_psd,
                                captions='Empty-Room Power Spectral Density '
                                         '(after filtering)',
                                section='Empty-Room')

    fname_report = bids_path.copy().update(suffix='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)
    import matplotlib.pyplot as plt  # nested import to help joblib
    plt.close('all')  # close all figures to save memory


def add_event_counts(*,
                     session: str,
                     report: mne.Report) -> None:
    try:
        df_events = count_events(BIDSPath(root=config.bids_root,
                                          session=session))
    except ValueError:
        logger.warn('Could not read events.')
        df_events = None

    if df_events is not None:
        css_classes = ('table', 'table-striped', 'table-borderless',
                       'table-hover')
        report.add_htmls_to_section(
            f'<div class="event-counts">\n'
            f'{df_events.to_html(classes=css_classes, border=0)}\n'
            f'</div>',
            captions='Event counts',
            section='events'
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
        report.add_custom_css(css)


# def add_epochs_drop_info(*,
#                          session: str,
#                          report: mne.Report) -> None:

#     epochs_fname = BIDSPath(session=session,
#                             task=config.get_task(),
#                             acquisition=config.acq,
#                             run=None,
#                             recording=config.rec,
#                             space=config.space,
#                             suffix='epo',
#                             extension='.fif',
#                             datatype=config.get_datatype(),
#                             root=config.deriv_root,
#                             check=False)

#     for subject in config.get_subjects():
#         fname_epochs = epochs_fname.update(subject=subject)
#         epochs = mne.read_epochs(fname_epochs)


def run_report_average(session: str) -> None:
    # Group report
    import matplotlib.pyplot as plt  # nested import to help joblib

    subject = 'average'
    evoked_fname = BIDSPath(subject=subject,
                            session=session,
                            task=config.get_task(),
                            acquisition=config.acq,
                            run=None,
                            recording=config.rec,
                            space=config.space,
                            suffix='ave',
                            extension='.fif',
                            datatype=config.get_datatype(),
                            root=config.deriv_root,
                            check=False)

    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=config.get_fs_subjects_dir())
    evokeds = mne.read_evokeds(evoked_fname)
    if config.analyze_channels:
        for evoked in evokeds:
            evoked.pick(config.analyze_channels)

    fs_subjects_dir = config.get_fs_subjects_dir()

    method = config.inverse_method
    inverse_str = method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph2fsaverage'

    conditions: List[Condition_T] = list(config.conditions)
    conditions.extend(config.contrasts)

    ###########################################################################
    #
    # Add events end epochs drop log stats.
    #
    add_event_counts(report=rep, session=session)

    ###########################################################################
    #
    # Visualize evoked responses.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in config.conditions:
            caption = f'Average: {condition}'
            section = 'Evoked'
        else:  # It's a contrast of two conditions.
            caption = f'Average Contrast: {condition[0]} – {condition[1]}'
            section = 'Contrast'

        fig = evoked.plot(spatial_colors=True, gfp=True, show=False)
        rep.add_figs_to_section(figs=fig, captions=caption,
                                comments=evoked.comment, section=section)

    ###########################################################################
    #
    # Visualize decoding results.
    #
    if config.decode:
        for contrast in config.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}-{cond_2}'.replace(op.sep, '')
            processing = f'{a_vs_b}+{config.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding_ = (evoked_fname.copy()
                               .update(processing=processing,
                                       suffix='decoding',
                                       extension='.mat'))
            decoding_data = loadmat(fname_decoding_)
            del fname_decoding_, processing, a_vs_b

            fig = plot_decoding_scores_gavg(decoding_data)
            caption = f'Time-by-time Decoding: {cond_1} ./. {cond_2}'
            comment = (f'Based on N={decoding_data["N"].squeeze()} subjects. '
                       f'Standard error and confidence interval of the mean '
                       f'were bootstrapped with {config.n_boot} resamples.')
            rep.add_figs_to_section(figs=fig, captions=caption,
                                    comments=comment,
                                    section='Decoding')
            del decoding_data, cond_1, cond_2, caption, comment

    ###########################################################################
    #
    # Visualize inverse solutions.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in config.conditions:
            caption = f'Average: {condition}'
            cond_str = config.sanitize_cond_name(condition)
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        section = 'Source'
        fname_stc_avg = evoked_fname.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}',
            extension=None)

        if op.exists(str(fname_stc_avg) + "-lh.stc"):
            stc = mne.read_source_estimate(fname_stc_avg, subject='fsaverage')
            _, peak_time = stc.get_peak()

            # Plot using 3d backend if available, and use Matplotlib
            # otherwise.
            if mne.viz.get_3d_backend() is not None:
                brain = stc.plot(views=['lat'], hemi='both',
                                 initial_time=peak_time, backend='pyvista',
                                 time_viewer=True,
                                 show_traces=True,
                                 subjects_dir=fs_subjects_dir)
                brain.toggle_interface()
                figs = brain._renderer.figure
                captions = caption
            else:
                fig_lh = plt.figure()
                fig_rh = plt.figure()

                brain_lh = stc.plot(views='lat', hemi='lh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_lh,
                                    subjects_dir=fs_subjects_dir)
                brain_rh = stc.plot(views='lat', hemi='rh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_rh,
                                    subjects_dir=fs_subjects_dir)
                figs = [brain_lh, brain_rh]
                captions = [f'{caption} - left',
                            f'{caption} - right']

            rep.add_figs_to_section(figs=figs, captions=captions,
                                    section='Sources')

            del peak_time

    fname_report = evoked_fname.copy().update(
        task=config.get_task(), suffix='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)

    msg = 'Completed Step 99: Create reports'
    logger.info(gen_log_message(step=99, message=msg))
    plt.close('all')  # close all figures to save memory


@failsafe_run(on_error=on_error)
def main():
    """Make reports."""
    msg = 'Running Step 99: Create reports'
    logger.info(gen_log_message(step=99, message=msg))

    parallel, run_func, _ = parallel_func(run_report, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    sessions = config.get_sessions()
    if not sessions:
        sessions = [None]

    for session in sessions:
        run_report_average(session)


if __name__ == '__main__':
    main()
