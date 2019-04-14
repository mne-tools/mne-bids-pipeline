"""
===============
06. Apply ICA
===============

Blinks and ECG artifacts are automatically detected and the corresponding ICA
components are removed from the data.
This relies on the ICAs computed in 05-run_ica.py
!! If you manually add components to remove (config.rejcomps_man),
make sure you did not re-run the ICA in the meantime. Otherwise (especially if
the random state was not set, or you used a different machine, the component
order might differ).

"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.report import Report

import numpy as np
import config


def apply_ica(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    # load epochs to reject ICA components
    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))
    epochs = mne.read_epochs(fname_in, preload=True)

    extension = '_cleaned-epo'
    fname_out = op.join(meg_subject_dir,
                        config.base_fname.format(**locals()))

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    # load first run of raw data for ecg /eog epochs
    raw_list = list()
    print("  Loading one run from raw data")
    extension = config.runs[0] + '_sss_raw'
    raw_fname_in = op.join(meg_subject_dir,
                           config.base_fname.format(**locals()))
    raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

    if config.eeg:
        ch_types = ['meg', 'eeg']
    else:
        ch_types = ['meg']

    for ch_type in ch_types:
        print(ch_type)
        picks = all_picks[ch_type]

        # Load ICA
        fname_ica = op.join(meg_subject_dir,
                            '{0}_{1}_{2}-ica.fif'.format(subject,
                                                         config.study_name,
                                                         ch_type))
        print('Reading ICA: ' + fname_ica)
        ica = read_ica(fname=fname_ica)

        pick_ecg = mne.pick_types(raw.info, meg=False, eeg=False,
                                  ecg=True, eog=False)

        # ECG
        # either needs an ecg channel, or avg of the mags (i.e. MEG data)
        if pick_ecg or ch_type == 'meg':

            picks_ecg = np.concatenate([picks, pick_ecg])

            # Create ecg epochs
            if ch_type == 'meg':
                reject = {'mag': config.reject['mag'],
                          'grad': config.reject['grad']}
            elif ch_type == 'eeg':
                reject = {'eeg': config.reject['eeg']}

            ecg_epochs = create_ecg_epochs(raw, picks=picks_ecg, reject=reject,
                                           baseline=(None, 0), tmin=-0.5,
                                           tmax=0.5)

            ecg_average = ecg_epochs.average()

            ecg_inds, scores = \
                ica.find_bads_ecg(ecg_epochs, method='ctps',
                                  threshold=config.ica_ctps_ecg_threshold)
            del ecg_epochs

            report_fname = \
                '{0}_{1}_{2}-reject_ica.html'.format(subject,
                                                     config.study_name,
                                                     ch_type)
            report_fname = op.join(meg_subject_dir, report_fname)
            report = Report(report_fname, verbose=False)

            # Plot r score
            report.add_figs_to_section(ica.plot_scores(scores,
                                                       exclude=ecg_inds,
                                                       show=config.plot),
                                       captions=ch_type.upper() + ' - ECG - ' +
                                       'R scores')

            # Plot source time course
            report.add_figs_to_section(ica.plot_sources(ecg_average,
                                                        exclude=ecg_inds,
                                                        show=config.plot),
                                       captions=ch_type.upper() + ' - ECG - ' +
                                       'Sources time course')

            # Plot source time course
            report.add_figs_to_section(ica.plot_overlay(ecg_average,
                                                        exclude=ecg_inds,
                                                        show=config.plot),
                                       captions=ch_type.upper() + ' - ECG - ' +
                                       'Corrections')

        else:
            # XXX : to check when EEG only is processed
            print('no ECG channel is present. Cannot automate ICAs component '
                  'detection for EOG!')

        # EOG
        pick_eog = mne.pick_types(raw.info, meg=False, eeg=False,
                                  ecg=False, eog=True)

        if pick_eog.any():
            print('using EOG channel')
            picks_eog = np.concatenate([picks, pick_eog])
            # Create eog epochs
            eog_epochs = create_eog_epochs(raw, picks=picks_eog, reject=None,
                                           baseline=(None, 0), tmin=-0.5,
                                           tmax=0.5)

            eog_average = eog_epochs.average()
            eog_inds, scores = ica.find_bads_eog(eog_epochs, threshold=3.0)
            del eog_epochs

            params = dict(exclude=eog_inds, show=config.plot)

            # Plot r score
            report.add_figs_to_section(ica.plot_scores(scores, **params),
                                       captions=ch_type.upper() + ' - EOG - ' +
                                       'R scores')

            # Plot source time course
            report.add_figs_to_section(ica.plot_sources(eog_average, **params),
                                       captions=ch_type.upper() + ' - EOG - ' +
                                       'Sources time course')

            # Plot source time course
            report.add_figs_to_section(ica.plot_overlay(eog_average, **params),
                                       captions=ch_type.upper() + ' - EOG - ' +
                                       'Corrections')

            report.save(report_fname, overwrite=True, open_browser=False)

        else:
            print('no EOG channel is present. Cannot automate ICAs component '
                  'detection for EOG!')

        ica_reject = (list(ecg_inds) + list(eog_inds) +
                      list(config.rejcomps_man[subject][ch_type]))

        # now reject the components
        print('Rejecting from %s: %s' % (ch_type, ica_reject))
        epochs = ica.apply(epochs, exclude=ica_reject)

        print('Saving cleaned epochs')
        epochs.save(fname_out)

        fig = ica.plot_overlay(raw, exclude=ica_reject, show=config.plot)
        report.add_figs_to_section(fig, captions=ch_type.upper() +
                                   ' - ALL(epochs) - Corrections')

        if config.plot:
            epochs.plot_image(combine='gfp', group_by='type', sigma=2.,
                              cmap="YlGnBu_r", show=config.plot)


parallel, run_func, _ = parallel_func(apply_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
