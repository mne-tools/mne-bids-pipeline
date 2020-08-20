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
import numpy as np

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def plot_events(subject, session, deriv_path):
    raws_filt = []
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path,
                                       kind=config.get_kind(),
                                       processing='filt',
                                       extension='.fif')

    for run in config.get_runs():
        fname = bids_basename.copy().update(run=run)
        raw_filt = mne.io.read_raw_fif(fname)
        raws_filt.append(raw_filt)
        del fname

    # Concatenate the filtered raws and extract the events.
    raw_filt_concat = mne.concatenate_raws(raws_filt)
    events, event_id = mne.events_from_annotations(raw=raw_filt_concat)
    fig = mne.viz.plot_events(events=events, event_id=event_id,
                              first_samp=raw_filt_concat.first_samp,
                              sfreq=raw_filt_concat.info['sfreq'],
                              show=False)
    return fig


def plot_er_psd(subject, session):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       acquisition=config.acq,
                                       run=None,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path,
                                       kind=config.get_kind(),
                                       task='noise',
                                       processing='filt',
                                       extension='.fif')

    extra_params = dict()
    if not config.use_maxwell_filter and config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    raw_er_filtered = mne.io.read_raw_fif(bids_basename, preload=True,
                                          **extra_params)

    fmax = 1.5 * config.h_freq if config.h_freq is not None else np.inf
    fig = raw_er_filtered.plot_psd(fmax=fmax, show=False)
    return fig


def plot_auto_scores(subject, session):
    """Plot automated bad channel detection scores.
    """
    import json_tricks

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    fname_scores = make_bids_basename(subject=subject,
                                      session=session,
                                      task=config.get_task(),
                                      acquisition=config.acq,
                                      run=None,
                                      processing=config.proc,
                                      recording=config.rec,
                                      space=config.space,
                                      prefix=deriv_path)

    fname_scores.update(kind='scores', extension='.json')

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


def run_report(subject, session=None):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path)

    fname_ave = bids_basename.copy().update(kind='ave', extension='.fif')
    fname_trans = bids_basename.copy().update(kind='trans', extension='.fif')
    subjects_dir = config.get_fs_subjects_dir()
    params = dict(info_fname=fname_ave, raw_psd=True)

    if op.exists(fname_trans):
        params['subject'] = subject
        params['subjects_dir'] = subjects_dir

    rep = mne.Report(**params)
    rep.parse_folder(deriv_path, verbose=True)

    # Visualize automated noisy channel detection.
    if config.find_noisy_channels_meg:
        figs, captions = plot_auto_scores(subject=subject, session=session)
        rep.add_figs_to_section(figs=figs,
                                captions=captions,
                                section='Data Quality')

    # Visualize events.
    events_fig = plot_events(subject=subject, session=session,
                             deriv_path=deriv_path)
    rep.add_figs_to_section(figs=events_fig,
                            captions='Events in filtered continuous data',
                            section='Events')

    conditions = config.conditions.copy()
    conditions.extend(config.contrasts)
    evokeds = mne.read_evokeds(fname_ave)

    ###########################################################################
    #
    # Visualize evoked responses.
    #
    for condition, evoked in zip(conditions, evokeds):
        if condition in config.conditions:
            caption = f'Condition: {condition}'
            section = 'Evoked'
        else:  # It's a contrast of two conditions.
            caption = f'Contrast: {condition[0]} – {condition[1]}'
            section = 'Contrast'

        fig = evoked.plot(show=False, gfp=True, spatial_colors=True)
        rep.add_figs_to_section(figs=fig, captions=caption,
                                comments=evoked.comment, section=section)

    ###########################################################################
    #
    # Visualize the coregistration & inverse solutions.
    #
    if op.exists(fname_trans):
        # We can only plot the coregistration if we have a valid 3d backend.
        if mne.viz.get_3d_backend() is not None:
            fig = mne.viz.plot_alignment(evoked.info, fname_trans,
                                         subject=subject,
                                         subjects_dir=subjects_dir,
                                         meg=True, dig=True, eeg=True)
            rep.add_figs_to_section(figs=fig, captions='Coregistration',
                                    section='Coregistration')
        else:
            msg = ('Cannot render sensor alignment (coregistration) because '
                   'no usable 3d backend was found.')
            logger.warning(gen_log_message(message=msg, step=99,
                                           subject=subject, session=session))

        for evoked in evokeds:
            msg = f'Rendering inverse solution for {evoked.comment} …'
            logger.info(gen_log_message(message=msg, step=99,
                                        subject=subject, session=session))

            if condition in config.conditions:
                caption = f'Condition: {condition}'
            else:  # It's a contrast of two conditions.
                # XXX Will change once we process contrasts here too
                continue

            method = config.inverse_method
            cond_str = evoked.comment.replace(op.sep, '').replace('_', '')
            inverse_str = '%s' % method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.

            fname_stc = bids_basename.copy().update(
                kind=f'{cond_str}+{inverse_str}+{hemi_str}')

            if op.exists(str(fname_stc) + "-lh.stc"):
                stc = mne.read_source_estimate(fname_stc, subject)
                _, peak_time = stc.get_peak()

                # Plot using 3d backend if available, and use Matplotlib
                # otherwise.
                if mne.viz.get_3d_backend() is not None:
                    brain = stc.plot(views=['lat'], hemi='both',
                                     initial_time=peak_time,  backend='mayavi')
                    figs = brain._figures[0]
                    comments = evoked.comment
                    captions = caption
                else:
                    import matplotlib.pyplot as plt
                    fig_lh = plt.figure()
                    fig_rh = plt.figure()

                    brain_lh = stc.plot(views='lat', hemi='lh',
                                        initial_time=peak_time,
                                        backend='matplotlib',
                                        subjects_dir=subjects_dir,
                                        figure=fig_lh)
                    brain_rh = stc.plot(views='lat', hemi='rh',
                                        initial_time=peak_time,
                                        subjects_dir=subjects_dir,
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

    if config.noise_cov == 'emptyroom':
        fig_er_psd = plot_er_psd(subject=subject, session=session)
        rep.add_figs_to_section(figs=fig_er_psd,
                                captions='Empty-Room Power Spectral Density '
                                         '(after filtering)',
                                section='Empty-Room')

    fname_report = bids_basename.copy().update(
        kind='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)


@failsafe_run(on_error=on_error)
def main():
    """Make reports."""
    msg = 'Running Step 99: Create reports'
    logger.info(gen_log_message(step=99, message=msg))

    parallel, run_func, _ = parallel_func(run_report, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    # Group report
    subject = 'average'
    # XXX to fix
    if config.get_sessions():
        session = config.get_sessions()[0]
    else:
        session = None

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())
    evoked_fname = make_bids_basename(subject=subject,
                                      session=session,
                                      task=config.get_task(),
                                      acquisition=config.acq,
                                      run=None,
                                      recording=config.rec,
                                      space=config.space,
                                      prefix=deriv_path,
                                      extension='.fif')
    evoked_fname.update(kind='ave')

    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=config.get_fs_subjects_dir())
    evokeds = mne.read_evokeds(evoked_fname)
    deriv_path = config.deriv_root
    subjects_dir = config.get_fs_subjects_dir()

    method = config.inverse_method
    inverse_str = method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph2fsaverage'

    conditions = config.conditions.copy()
    conditions.extend(config.contrasts)

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

        fig = evoked.plot(show=False, gfp=True, spatial_colors=True)
        fig = evoked.plot(spatial_colors=True, gfp=True, show=False)
        rep.add_figs_to_section(figs=fig, captions=caption,
                                comments=evoked.comment, section=section)

    ###########################################################################
    #
    # Visualize inverse solutions.
    #

    for condition, evoked in zip(conditions, evokeds):
        if condition in config.conditions:
            caption = f'Average: {condition}'
            cond_str = condition.replace(op.sep, '').replace('_', '')
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        section = 'Source'
        fname_stc_avg = evoked_fname.copy().update(
            kind=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}',
            extension=None)

        if op.exists(str(fname_stc_avg) + "-lh.stc"):
            stc = mne.read_source_estimate(fname_stc_avg, subject='fsaverage')
            _, peak_time = stc.get_peak()

            # Plot using 3d backend if available, and use Matplotlib
            # otherwise.
            if mne.viz.get_3d_backend() is not None:
                brain = stc.plot(views=['lat'], hemi='both',
                                 initial_time=peak_time, backend='mayavi',
                                 subjects_dir=subjects_dir)
                figs = brain._figures[0]
                captions = caption
            else:
                import matplotlib.pyplot as plt
                fig_lh = plt.figure()
                fig_rh = plt.figure()

                brain_lh = stc.plot(views='lat', hemi='lh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_lh,
                                    subjects_dir=subjects_dir)
                brain_rh = stc.plot(views='lat', hemi='rh',
                                    initial_time=peak_time,
                                    backend='matplotlib', figure=fig_rh,
                                    subjects_dir=subjects_dir)
                figs = [brain_lh, brain_rh]
                captions = [f'{caption} - left',
                            f'{caption} - right']

            rep.add_figs_to_section(figs=figs, captions=captions,
                                    section='Sources')

            del peak_time

    fname_report = evoked_fname.copy().update(
        task=config.get_task(), kind='report', extension='.html')
    rep.save(fname=fname_report, open_browser=False, overwrite=True)

    msg = 'Completed Step 99: Create reports'
    logger.info(gen_log_message(step=99, message=msg))


if __name__ == '__main__':
    main()
