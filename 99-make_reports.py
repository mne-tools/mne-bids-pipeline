"""
================
99. Make reports
================

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

from mayavi import mlab
import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_report(subject):
    print("Processing %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '-ave'
    fname_ave = op.join(meg_subject_dir,
                        config.base_fname.format(**locals()))
    fname_trans = op.join(meg_subject_dir,
                          config.base_fname_trans.format(**locals()))
    subjects_dir = config.subjects_dir
    if not op.exists(fname_trans):
        subject = None
        subjects_dir = None

    rep = mne.Report(info_fname=fname_ave, subject=subject,
                     subjects_dir=subjects_dir)
    rep.parse_folder(meg_subject_dir)

    evokeds = mne.read_evokeds(fname_ave)

    figs = list()
    captions = list()

    for evoked in evokeds:
        fig = evoked.plot(spatial_colors=True, show=False, gfp=True)
        figs.append(fig)
        captions.append(evoked.comment)

    if op.exists(fname_trans):
        mne.viz.plot_alignment(evoked.info, fname_trans, subject=subject,
                               subjects_dir=config.subjects_dir, meg=True,
                               dig=True, eeg=True)
        fig = mlab.gcf()
        figs.append(fig)
        captions.append('Coregistration')

        rep.add_figs_to_section(figs, captions)
        for evoked in evokeds:
            fname = op.join(meg_subject_dir, 'mne_dSPM_inverse-%s'
                            % evoked.comment)
            stc = mne.read_source_estimate(fname, subject)
            brain = stc.plot(views=['ven'], hemi='both')

            brain.set_data_time_index(112)

            fig = mlab.gcf()
            rep._add_figs_to_section(fig, evoked.condition)

    rep.save(fname=op.join(meg_subject_dir, 'report_%s.html' % subject),
             open_browser=False, overwrite=True)


parallel, run_func, _ = parallel_func(run_report, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)

# Group report
evoked_fname = op.join(config.meg_dir,
                       '%s_grand_average-ave.fif' % config.study_name)
rep = mne.Report(info_fname=evoked_fname, subject='fsaverage',
                 subjects_dir=config.subjects_dir)
evokeds = mne.read_evokeds(evoked_fname)

for evoked, condition in zip(evokeds, config.conditions):
    rep.add_figs_to_section(evoked.plot(spatial_colors=True, gfp=True,
                                        show=False),
                            'Average %s' % condition)

    stc_fname = op.join(config.meg_dir, 'average_dSPM-%s' % condition)
    if op.exists(stc_fname + "-lh.stc"):
        stc = mne.read_source_estimate(stc_fname, subject='fsaverage')
        brain = stc.plot(views=['lat'], hemi='both', subject='fsaverage',
                         subjects_dir=config.subjects_dir)
        brain.set_data_time_index(165)

        fig = mlab.gcf()
        rep.add_figs_to_section(fig, 'Average %s' % condition)

rep.save(fname=op.join(config.meg_dir, 'report_average.html'),
         open_browser=False, overwrite=True)
