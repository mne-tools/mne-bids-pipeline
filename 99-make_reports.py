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


def run_report(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
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
    subjects_dir = config.subjects_dir
    if not op.exists(fname_trans):
        subject = None
        subjects_dir = None

    rep = mne.Report(info_fname=fname_ave, subject=subject,
                     subjects_dir=subjects_dir)
    rep.parse_folder(fpath_deriv, verbose=True)

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
                                     subjects_dir=config.subjects_dir,
                                     meg=True, dig=True, eeg=True)
        rep.add_figs_to_section(fig, 'Coregistration')

        for evoked in evokeds:
            fname = op.join(fpath_deriv, 'mne_dSPM_inverse-%s'
                            % evoked.comment.replace(op.sep, ''))
            stc = mne.read_source_estimate(fname, subject)
            brain = stc.plot(views=['lat'], hemi='both')

            brain.set_data_time_index(112)
            fig = brain._figures[0]

            rep.add_figs_to_section(fig, evoked.condition)

    rep.save(fname=op.join(fpath_deriv, 'report.html'),
             open_browser=False, overwrite=True)


def main():
    """Make reports."""
    parallel, run_func, _ = parallel_func(run_report, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))

    # Group report
    evoked_fname = op.join(config.bids_root, 'derivatives',
                           config.PIPELINE_NAME,
                           '%s_grand_average-ave.fif' % config.study_name)
    rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                     subjects_dir=config.subjects_dir)
    evokeds = mne.read_evokeds(evoked_fname)

    for evoked, condition in zip(evokeds, config.conditions):
        rep.add_figs_to_section(evoked.plot(spatial_colors=True, gfp=True,
                                            show=False),
                                'Average %s' % condition)

        stc_fname = op.join(config.bids_root, 'derivatives',
                            config.PIPELINE_NAME,
                            'average_dSPM-%s' % condition.replace(op.sep, ''))
        if op.exists(stc_fname + "-lh.stc"):
            stc = mne.read_source_estimate(stc_fname, subject='fsaverage')
            brain = stc.plot(views=['lat'], hemi='both', subject='fsaverage',
                             subjects_dir=config.subjects_dir)
            brain.set_data_time_index(165)
            fig = brain._figures[0]

            rep.add_figs_to_section(fig, 'Average %s' % condition)

    rep.save(fname=op.join(config.bids_root, 'derivatives',
                           config.PIPELINE_NAME, 'report_average.html'),
             open_browser=False, overwrite=True)


if __name__ == '__main__':
    main()
