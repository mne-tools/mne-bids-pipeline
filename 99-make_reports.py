"""
================
99. Make reports
================

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


def plot_events(subject, session, fpath_deriv):
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
        fname = op.join(fpath_deriv, bids_basename + '_filt_raw.fif')
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


def run_report(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
    fname_ave = \
        op.join(fpath_deriv, bids_basename + '-ave.fif')
    fname_trans = \
        op.join(fpath_deriv, 'sub-{}'.format(subject) + '-trans.fif')
    subjects_dir = config.get_subjects_dir()
    if op.exists(fname_trans):
        rep = mne.Report(info_fname=fname_ave, subject=subject,
                         subjects_dir=subjects_dir)
    else:
        rep = mne.Report(info_fname=fname_ave)

    rep.parse_folder(fpath_deriv, verbose=True)

    # Visualize events.
    events_fig = plot_events(subject=subject, session=session,
                             fpath_deriv=fpath_deriv)
    rep.add_figs_to_section([events_fig],
                            ['Events in filtered continuous data'])

    # Visualize evoked responses.
    evokeds = mne.read_evokeds(fname_ave)
    figs = list()
    captions = list()

    for evoked in evokeds:
        # fig = evoked.plot(spatial_colors=True, show=False, gfp=True)
        fig = evoked.plot(show=False, gfp=True)
        figs.append(fig)
        captions.append(evoked.comment)

    rep.add_figs_to_section(figs, captions)

    if op.exists(fname_trans):
        fig = mne.viz.plot_alignment(evoked.info, fname_trans,
                                     subject=subject,
                                     subjects_dir=config.get_subjects_dir(),
                                     meg=True, dig=True, eeg=True)
        rep.add_figs_to_section(fig, 'Coregistration')

        for evoked in evokeds:
            method = config.inverse_method
            cond_str = 'cond-%s' % evoked.comment.replace(op.sep, '')
            inverse_str = 'inverse-%s' % method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
            fname_stc = op.join(fpath_deriv, '_'.join([bids_basename, cond_str,
                                                       inverse_str, hemi_str]))

            if op.exists(fname_stc + "-lh.stc"):
                stc = mne.read_source_estimate(fname_stc, subject)
                _, peak_time = stc.get_peak()
                brain = stc.plot(views=['lat'], hemi='both',
                                 initial_time=peak_time)
                fig = brain._figures[0]
                rep.add_figs_to_section(fig, evoked.condition)

                del peak_time

    if config.get_task():
        task_str = '_task-%s' % config.get_task()
    else:
        task_str = ''

    fname_report = op.join(fpath_deriv, 'report%s.html' % task_str)
    rep.save(fname=fname_report, open_browser=False, overwrite=True)


def main():
    """Make reports."""
    parallel, run_func, _ = parallel_func(run_report, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    # Group report
    evoked_fname = op.join(config.bids_root, 'derivatives',
                           config.PIPELINE_NAME,
                           '%s_grand_average-ave.fif' % config.study_name)
    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=config.get_subjects_dir())
    evokeds = mne.read_evokeds(evoked_fname)

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME)

    bids_basename = make_bids_basename(task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    for evoked, condition in zip(evokeds, config.conditions):
        rep.add_figs_to_section(evoked.plot(spatial_colors=True, gfp=True,
                                            show=False),
                                'Average %s' % condition)

        method = config.inverse_method
        cond_str = 'cond-%s' % condition.replace(op.sep, '')
        inverse_str = 'inverse-%s' % method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph-fsaverage'

        fname_stc_avg = op.join(fpath_deriv, '_'.join(['average',
                                                       bids_basename, cond_str,
                                                       inverse_str, morph_str,
                                                       hemi_str]))

        if op.exists(fname_stc_avg + "-lh.stc"):
            stc = mne.read_source_estimate(fname_stc_avg, subject='fsaverage')
            _, peak_time = stc.get_peak()
            brain = stc.plot(views=['lat'], hemi='both', subject='fsaverage',
                             subjects_dir=config.get_subjects_dir(),
                             initial_time=peak_time)

            fig = brain._figures[0]
            rep.add_figs_to_section(fig, 'Average %s' % condition)

            del peak_time

    if config.get_task():
        task_str = '_task-%s' % config.get_task()
    else:
        task_str = ''

    fname_report = op.join(fpath_deriv, 'report_average%s.html' % task_str)
    rep.save(fname=fname_report, open_browser=False, overwrite=True)


if __name__ == '__main__':
    main()
