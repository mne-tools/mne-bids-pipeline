"""
================
99. Make reports
================

Builds an HTML report for each subject containing all the relevant analysis
plots.
"""

from mayavi import mlab
import os.path as op

from mne import Report
import mne

import config


def make_report(subject_id):
    subject = "sub%03d" % subject_id
    print("processing %s" % subject)

    meg_path = op.join(meg_dir, subject)
    ave_fname = op.join(meg_path, "%s-ave.fif" % subject)

    rep = Report(info_fname=ave_fname, subject=subject,
                 subjects_dir=subjects_dir)
    rep.parse_folder(meg_path)

    evokeds = mne.read_evokeds(ave_fname)

    figs = list()
    captions = list()

    for evoked in evokeds:
        fig = fam.plot(spatial_colors=True, show=False, gfp=True)
        figs.append(fig)
        captions.append(evoked.condition)

    fname_trans = op.join(study_path, 'MEG', 'subject', '%s-trans.fif' % subject)
    mne.viz.plot_trans(fam.info, fname_trans, subject=subject,
                       subjects_dir=subjects_dir, meg_sensors=True,
                       eeg_sensors=True)
    fig = mlab.gcf()
    figs.append(fig)
    captions.append('Coregistration')

    rep.add_figs_to_section(figs, captions)
    for evoked in evokeds:
        fname = op.join(meg_path, 'mne_dSPM_inverse-%s'
                        % evoked.condition)
        stc = mne.read_source_estimate(fname, subject)
        brain = stc.plot(views=['ven'], hemi='both')

        brain.set_data_time_index(112)

        fig = mlab.gcf()
        rep._add_figs_to_section(fig, cond)

    rep.save(fname=op.join(meg_dir, 'report%s.html' % subject),
             open_browser=False, overwrite=True)


# # Group report
# faces_fname = op.join(meg_dir, 'eeg_faces_highpass-%sHz-ave.fif' % l_freq)
# rep = Report(info_fname=faces_fname, subject='fsaverage',
#              subjects_dir=subjects_dir)
# faces = mne.read_evokeds(faces_fname)[0]
# rep.add_figs_to_section(faces.plot(spatial_colors=True, gfp=True, show=False),
#                         'Average faces')

# scrambled = mne.read_evokeds(op.join(meg_dir, 'eeg_scrambled-ave.fif'))[0]
# rep.add_figs_to_section(scrambled.plot(spatial_colors=True, gfp=True,
#                                        show=False), 'Average scrambled')

# fname = op.join(meg_dir, 'contrast-average_highpass-%sHz' % l_freq)
# stc = mne.read_source_estimate(fname, subject='fsaverage')
# brain = stc.plot(views=['ven'], hemi='both', subject='fsaverage',
#                  subjects_dir=subjects_dir)
# brain.set_data_time_index(165)

# fig = mlab.gcf()
# rep.add_figs_to_section(fig, 'Average faces - scrambled')

rep.save(fname=op.join(meg_dir, 'report_average.html'),
         open_browser=False, overwrite=True)
