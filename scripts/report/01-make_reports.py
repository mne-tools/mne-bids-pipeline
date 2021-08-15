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
from typing import Dict, Any, Tuple, Union, Optional

import numpy as np
from scipy.io import loadmat
import matplotlib

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne_bids import BIDSPath
from mne_bids.stats import count_events

import config
from config import gen_log_kwargs, on_error, failsafe_run

matplotlib.use('Agg')  # do not open any window  # noqa

logger = logging.getLogger('mne-bids-pipeline')

Condition_T = Union[str, Tuple[str]]


def plot_events(cfg, subject, session):
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
    fig = mne.viz.plot_events(events=events, event_id=event_id,
                              first_samp=raw_filt_concat.first_samp,
                              sfreq=raw_filt_concat.info['sfreq'],
                              show=False)
    return fig


def plot_er_psd(cfg, subject, session):
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

    raw_er_filtered = mne.io.read_raw_fif(raw_fname, preload=True)

    fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
    fig = raw_er_filtered.plot_psd(fmax=fmax, show=False)
    return fig


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


@failsafe_run(on_error=on_error, script_path=__file__)
def run_report(*, cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    fname_ave = bids_path.copy().update(suffix='ave')
    fname_epo = bids_path.copy().update(suffix='epo')
    if cfg.use_template_mri:
        fname_trans = 'fsaverage'
        has_trans = True
    else:
        fname_trans = bids_path.copy().update(suffix='trans')
        has_trans = op.exists(fname_trans)

    fname_epo = bids_path.copy().update(processing='clean', suffix='epo')
    fname_ica = bids_path.copy().update(suffix='ica')
    fname_decoding = fname_epo.copy().update(processing=None,
                                             suffix='decoding',
                                             extension='.mat')
    fname_tfr_pow = bids_path.copy().update(suffix='power+condition+tfr',
                                            extension='.h5')

    title = f'sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    params: Dict[str, Any] = dict(info_fname=fname_epo, raw_psd=True,
                                  subject=cfg.fs_subject, title=title)
    if has_trans:
        params['subjects_dir'] = cfg.fs_subjects_dir

    rep = mne.Report(**params)
    rep_kwargs: Dict[str, Any] = dict(data_path=fname_ave.fpath.parent,
                                      verbose=False)
    if not has_trans:
        rep_kwargs['render_bem'] = False

    if cfg.task is not None:
        rep_kwargs['pattern'] = f'*_task-{cfg.task}*'
    if mne.viz.get_3d_backend() is not None:
        with mne.viz.use_3d_backend('pyvistaqt'):
            rep.parse_folder(**rep_kwargs)
    else:
        rep.parse_folder(**rep_kwargs)

    # Visualize automated noisy channel detection.
    if cfg.find_noisy_channels_meg:
        figs, captions = plot_auto_scores(cfg=cfg, subject=subject,
                                          session=session)
        rep.add_figs_to_section(figs=figs,
                                captions=captions,
                                section='Data Quality')

    # Visualize events.
    if cfg.task.lower() != 'rest':
        events_fig = plot_events(cfg=cfg, subject=subject, session=session)
        rep.add_figs_to_section(figs=events_fig,
                                captions='Events in filtered continuous data',
                                section='Events')

    ###########################################################################
    #
    # Visualize effect of ICA artifact rejection.
    #
    if cfg.spatial_filter == 'ica':
        epochs = mne.read_epochs(fname_epo)
        ica = mne.preprocessing.read_ica(fname_ica)
        fig = ica.plot_overlay(epochs.average(), show=False)
        rep.add_figs_to_section(
            fig,
            captions=f'Evoked response (across all epochs) '
                     f'before and after ICA '
                     f'({len(ica.exclude)} ICs removed)',
            section='ICA'
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

    for condition in conditions:
        cond = config.sanitize_cond_name(condition)
        fname_tfr_pow_cond = str(fname_tfr_pow.copy()).replace("+condition+",
                                                               f"+{cond}+")
        power = mne.time_frequency.read_tfrs(fname_tfr_pow_cond)
        fig = power[0].plot_topo(show=False, fig_facecolor='w', font_color='k',
                                 border='k')
        rep.add_figs_to_section(figs=fig, captions=f"TFR Power: {condition}",
                                section="TFR")

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

    for condition, evoked in zip(conditions, evokeds):
        if cfg.analyze_channels:
            evoked.pick(cfg.analyze_channels)

        if condition in cfg.conditions:
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
    if cfg.decode:
        epochs = mne.read_epochs(fname_epo)

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
                metric=cfg.decoding_metric)

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
    if has_trans:
        evokeds = mne.read_evokeds(fname_ave)

        # Omit our custom coreg plot here – this is now handled through
        # parse_folder() automatically. Keep the following code around for
        # future reference.
        #
        # # We can only plot the coregistration if we have a valid 3d backend.
        # if mne.viz.get_3d_backend() is not None:
        #     fig = mne.viz.plot_alignment(evoked.info, fname_trans,
        #                                  subject=cfg.fs_subject,
        #                                  subjects_dir=cfg.fs_subjects_dir,
        #                                  meg=True, dig=True, eeg=True)
        #     rep.add_figs_to_section(figs=fig, captions='Coregistration',
        #                             section='Coregistration')
        # else:
        #     msg = ('Cannot render sensor alignment (coregistration) because '
        #            'no usable 3d backend was found.')
        #     logger.warning(gen_log_message(message=msg,
        #                                    subject=subject, session=session))

        for condition, evoked in zip(conditions, evokeds):
            msg = f'Rendering inverse solution for {evoked.comment} …'
            logger.info(**gen_log_kwargs(message=msg,
                                         subject=subject, session=session))

            if condition in cfg.conditions:
                full_condition = config.sanitize_cond_name(evoked.comment)
                caption = f'Condition: {full_condition}'
                del full_condition
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

            if op.exists(str(fname_stc) + "-lh.stc"):
                stc = mne.read_source_estimate(fname_stc,
                                               subject=cfg.fs_subject)
                _, peak_time = stc.get_peak()

                # Plot using 3d backend if available, and use Matplotlib
                # otherwise.
                import matplotlib.pyplot as plt

                if mne.viz.get_3d_backend() is not None:
                    brain = stc.plot(views=['lat'], hemi='split',
                                     initial_time=peak_time, backend='pyvista',
                                     time_viewer=True,
                                     subjects_dir=cfg.fs_subjects_dir)
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
                                        subjects_dir=cfg.fs_subjects_dir,
                                        figure=fig_lh)
                    brain_rh = stc.plot(views='lat', hemi='rh',
                                        initial_time=peak_time,
                                        subjects_dir=cfg.fs_subjects_dir,
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

    if cfg.process_er:
        fig_er_psd = plot_er_psd(cfg=cfg, subject=subject, session=session)
        rep.add_figs_to_section(figs=fig_er_psd,
                                captions='Empty-Room Power Spectral Density '
                                         '(after filtering)',
                                section='Empty-Room')

    fname_report = bids_path.copy().update(suffix='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)
    import matplotlib.pyplot as plt  # nested import to help joblib
    plt.close('all')  # close all figures to save memory


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
#                             task=cfg.task,
#                             acquisition=cfg.acq,
#                             run=None,
#                             recording=cfg.rec,
#                             space=cfg.space,
#                             suffix='epo',
#                             extension='.fif',
#                             datatype=cfg.datatype,
#                             root=cfg.deriv_root,
#                             check=False)

#     for subject in config.get_subjects():
#         fname_epochs = epochs_fname.update(subject=subject)
#         epochs = mne.read_epochs(fname_epochs)


@failsafe_run(on_error=on_error, script_path=__file__)
def run_report_average(*, cfg, subject: str, session: str) -> None:
    # Group report
    import matplotlib.pyplot as plt  # nested import to help joblib

    evoked_fname = BIDSPath(subject=subject,
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
                            check=False)

    title = f'sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=cfg.fs_subjects_dir,
                     title=title)
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
    # Add events end epochs drop log stats.
    #
    add_event_counts(cfg=cfg, report=rep, session=session)

    #######################################################################
    #
    # Visualize evoked responses.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in cfg.conditions:
            caption = f'Average: {condition}'
            section = 'Evoked'
        else:  # It's a contrast of two conditions.
            caption = f'Average Contrast: {condition[0]} – {condition[1]}'
            section = 'Contrast'

        fig = evoked.plot(spatial_colors=True, gfp=True, show=False)
        rep.add_figs_to_section(figs=fig, captions=caption,
                                comments=evoked.comment, section=section)

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
            caption = f'Time-by-time Decoding: {cond_1} ./. {cond_2}'
            comment = (f'Based on N={decoding_data["N"].squeeze()} '
                       f'subjects. Standard error and confidence interval '
                       f'of the mean were bootstrapped with {cfg.n_boot} '
                       f'resamples.')
            rep.add_figs_to_section(figs=fig, captions=caption,
                                    comments=comment,
                                    section='Decoding')
            del decoding_data, cond_1, cond_2, caption, comment

    #######################################################################
    #
    # Visualize inverse solutions.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in cfg.conditions:
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
            stc = mne.read_source_estimate(fname_stc_avg,
                                           subject='fsaverage')
            _, peak_time = stc.get_peak()

            # Plot using 3d backend if available, and use Matplotlib
            # otherwise.
            if mne.viz.get_3d_backend() is not None:
                brain = stc.plot(views=['lat'], hemi='both',
                                 initial_time=peak_time, backend='pyvista',
                                 time_viewer=True,
                                 show_traces=True,
                                 subjects_dir=cfg.fs_subjects_dir)
                brain.toggle_interface()
                figs = brain._renderer.figure
                captions = caption
            else:
                fig_lh = plt.figure()
                fig_rh = plt.figure()

                brain_lh = stc.plot(views='lat', hemi='lh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_lh,
                                    subjects_dir=cfg.fs_subjects_dir)
                brain_rh = stc.plot(views='lat', hemi='rh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_rh,
                                    subjects_dir=cfg.fs_subjects_dir)
                figs = [brain_lh, brain_rh]
                captions = [f'{caption} - left',
                            f'{caption} - right']

            rep.add_figs_to_section(figs=figs, captions=captions,
                                    section='Sources')

            del peak_time

    fname_report = evoked_fname.copy().update(
        task=cfg.task, suffix='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)

    plt.close('all')  # close all figures to save memory


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
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

    cfg = BunchConst(
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
    )
    return cfg


def main():
    """Make reports."""
    parallel, run_func, _ = parallel_func(run_report,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(subject=subject), subject=subject,
                 session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
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
        run_report_average(cfg=get_config(subject='average'),
                           subject='average',
                           session=session)


if __name__ == '__main__':
    main()
