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

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def plot_events(subject, session, deriv_path):
    raws_filt = []
    for run in config.get_runs():
        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.get_task(),
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space)
        fname = op.join(deriv_path, bids_basename + '_filt_raw.fif')
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
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    raw_er_filtered_fname = op.join(deriv_path,
                                    f'{bids_basename}_emptyroom_filt_raw.fif')

    extra_params = dict()
    if not config.use_maxwell_filter and config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    raw_er_filtered = mne.io.read_raw_fif(raw_er_filtered_fname, preload=True,
                                          **extra_params)
    fig = raw_er_filtered.plot_psd(show=False)
    return fig


def run_report(subject, session=None):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    fname_ave = op.join(deriv_path, bids_basename + '-ave.fif')
    fname_trans = op.join(deriv_path, 'sub-{}'.format(subject) + '-trans.fif')
    subjects_dir = config.get_fs_subjects_dir()
    if op.exists(fname_trans):
        rep = mne.Report(info_fname=fname_ave, subject=subject,
                         subjects_dir=subjects_dir)
    else:
        rep = mne.Report(info_fname=fname_ave)

    rep.parse_folder(deriv_path, verbose=True)

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
            logger.warn(gen_log_message(message=msg, step=99,
                                        subject=subject, session=session))

        for evoked in evokeds:
            msg = 'Rendering inverse solution for {evoked.comment} …'
            logger.info(gen_log_message(message=msg, step=99,
                                        subject=subject, session=session))

            if condition in config.conditions:
                caption = f'Condition: {condition}'
            else:  # It's a contrast of two conditions.
                # XXX Will change once we process contrasts here too
                continue

            method = config.inverse_method
            cond_str = 'cond-%s' % evoked.comment.replace(op.sep, '')
            inverse_str = 'inverse-%s' % method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
            fname_stc = op.join(deriv_path, '_'.join([bids_basename, cond_str,
                                                      inverse_str, hemi_str]))

            if op.exists(fname_stc + "-lh.stc"):
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

    if config.get_task():
        task_str = '_task-%s' % config.get_task()
    else:
        task_str = ''

    fname_report = op.join(deriv_path, 'report%s.html' % task_str)
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
    evoked_fname = op.join(config.bids_root, 'derivatives',
                           config.PIPELINE_NAME,
                           '%s_grand_average-ave.fif' % config.study_name)
    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=config.get_fs_subjects_dir())
    evokeds = mne.read_evokeds(evoked_fname)
    deriv_path = config.deriv_root
    subjects_dir = config.get_fs_subjects_dir()
    bids_basename = make_bids_basename(task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    method = config.inverse_method
    inverse_str = 'inverse-%s' % method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph-fsaverage'

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
            cond_str = 'cond-%s' % condition.replace(op.sep, '')
        else:  # It's a contrast of two conditions.
            # XXX Will change once we process contrasts here too
            continue

        section = 'Source'
        fname_stc_avg = op.join(deriv_path, '_'.join(['average',
                                                      bids_basename, cond_str,
                                                      inverse_str, morph_str,
                                                      hemi_str]))

        if op.exists(fname_stc_avg + "-lh.stc"):
            stc = mne.read_source_estimate(fname_stc_avg, subject='fsaverage')
            _, peak_time = stc.get_peak()

            # Plot using 3d backend if available, and use Matplotlib
            # otherwise.
            if mne.viz.get_3d_backend() is not None:
                brain = stc.plot(views=['lat'], hemi='both',
                                 initial_time=peak_time,  backend='mayavi',
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

    if config.get_task():
        task_str = '_task-%s' % config.get_task()
    else:
        task_str = ''

    fname_report = op.join(deriv_path, 'report_average%s.html' % task_str)
    rep.save(fname=fname_report, open_browser=False, overwrite=True)

    msg = 'Completed Step 99: Create reports'
    logger.info(gen_log_message(step=99, message=msg))


if __name__ == '__main__':
    main()
