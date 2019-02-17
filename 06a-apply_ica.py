"""
===============
06. Evoked data
===============

The evoked data sets are created by averaging different conditions.
"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.report import Report

import numpy as np
import config


def run_evoked(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    # load epochs to reject ICA components
    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))
    epochs = mne.read_epochs(fname_in, preload=True)

    extension = 'cleaned-epo'
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
    all_picks = dict({'meg': picks_meg, 'eeg': picks_eeg})

    for ch_type in ['meg', 'eeg']:
        print(ch_type)
        picks = all_picks[ch_type]

        # Load ICA
        fname_ica = op.join(meg_subject_dir,
                            '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name,
                                                         ch_type))
        print('Reading ICA: ' + fname_ica)
        ica = read_ica(fname=fname_ica)

        # ECG
        if config.ecg_channel and ch_type == 'meg':
            print('using ECG channel')
            picks_ecg = np.concatenate([picks, mne.pick_types(raw.info, meg=False,
                                                              eeg=False, ecg=True,
                                                              eog=False)])
            # Create ecg epochs
            ecg_epochs = create_ecg_epochs(raw, picks=picks_ecg, reject=None,
                                           baseline=(None, 0), tmin=-0.5,
                                           tmax=0.5)

            ecg_average = ecg_epochs.average()
            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
            del ecg_epochs

            if config.plot:

                report_name = op.join(meg_subject_dir,
                                      '{0}_{1}_{2}-reject_ica.html'.format(subject, config.study_name,
                                                                           ch_type))
                report = Report(report_name, verbose=False)

                # Plot r score
                report.add_figs_to_section(ica.plot_scores(scores, exclude=ecg_inds),
                                           captions=ch_type.upper() + ' - ECG - '
                                           + 'R scores')

                # Plot source time course
                report.add_figs_to_section(ica.plot_sources(ecg_average, exclude=ecg_inds),
                                           captions=ch_type.upper() + ' - ECG - '
                                           + 'Sources time course')

                # Plot source time course
                report.add_figs_to_section(ica.plot_overlay(ecg_average, exclude=ecg_inds),
                                           captions=ch_type.upper() + ' - ECG - '
                                           + 'Corrections')

        else:
            print('no ECG channel!')

        # EOG
        if config.eog_channel:
            print('using EOG channel')
            picks_eog = np.concatenate([picks, mne.pick_types(raw.info, meg=False,
                                                              eeg=False, ecg=False,
                                                              eog=True)])
            # Create eog epochs
            eog_epochs = create_eog_epochs(raw, picks=picks_eog, reject=None,
                                           baseline=(None, 0), tmin=-0.5,
                                           tmax=0.5)

            eog_average = eog_epochs.average()
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
            del eog_epochs

            if config.plot:
                # Plot r score
                report.add_figs_to_section(ica.plot_scores(scores, exclude=eog_inds),
                                           captions=ch_type.upper() + ' - EOG - '
                                           + 'R scores')

                # Plot source time course
                report.add_figs_to_section(ica.plot_sources(eog_average, exclude=eog_inds),
                                           captions=ch_type.upper() + ' - EOG - '
                                           + 'Sources time course')

                # Plot source time course
                report.add_figs_to_section(ica.plot_overlay(eog_average, exclude=eog_inds),
                                           captions=ch_type.upper() + ' - EOG - '
                                           + 'Corrections')

                report.save(report_name, overwrite=True, open_browser=False)

        else:
            print('no EOG channel!')

        ica_reject = np.ndarray.tolist(np.concatenate([ecg_inds, eog_inds,
                                                       config.rejcomps_man[subject][ch_type]]))

        # now reject the components
        print('Rejecting from ' + ch_type + ': ' + str(ica_reject))
        epochs = ica.apply(epochs, exclude=ica_reject)

        print('Saving epochs')
        epochs.save(fname_out)

        if config.plot:
            report.add_figs_to_section(ica.plot_overlay(raw.copy(), exclude=ica_reject),
                                       captions=ch_type.upper() +
                                       ' - ALL(epochs) - ' + 'Corrections')


parallel, run_func, _ = parallel_func(run_evoked, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
