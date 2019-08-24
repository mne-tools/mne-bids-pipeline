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
import itertools

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.report import Report

from mne_bids import make_bids_basename

import numpy as np
import config


def apply_ica(subject, run, session):
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
    fname_in = \
        op.join(fpath_deriv, bids_basename + '-epo.fif')

    fname_out = \
        op.join(fpath_deriv, bids_basename + '_cleaned-epo.fif')

    # load epochs to reject ICA components
    epochs = mne.read_epochs(fname_in, preload=True)

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    # load first run of raw data for ecg /eog epochs
    print("  Loading one run from raw data")

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=config.runs[0],
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    if config.use_maxwell_filter:
        raw_fname_in = \
            op.join(fpath_deriv, bids_basename + '_sss_raw.fif')
    else:
        raw_fname_in = \
            op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

    raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

    for ch_type in config.ch_types:
        report = None
        print(ch_type)
        picks = all_picks[ch_type]

        # Load ICA
        fname_ica = \
            op.join(fpath_deriv, bids_basename + '_%s-ica.fif' % ch_type)

        print('Reading ICA: ' + fname_ica)
        ica = read_ica(fname=fname_ica)

        pick_ecg = mne.pick_types(raw.info, meg=False, eeg=False,
                                  ecg=True, eog=False)

        # ECG
        # either needs an ecg channel, or avg of the mags (i.e. MEG data)
        ecg_inds = list()
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
                op.join(fpath_deriv,
                        bids_basename + '_%s-reject_ica.html' % ch_type)

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
                  'detection for ECG!')

        # EOG
        pick_eog = mne.pick_types(raw.info, meg=False, eeg=False,
                                  ecg=False, eog=True)
        eog_inds = list()
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

        if report is not None:
            fig = ica.plot_overlay(raw, exclude=ica_reject, show=config.plot)
            report.add_figs_to_section(fig, captions=ch_type.upper() +
                                       ' - ALL(epochs) - Corrections')

        if config.plot:
            epochs.plot_image(combine='gfp', group_by='type', sigma=2.,
                              cmap="YlGnBu_r", show=config.plot)


def main():
    """Apply ICA."""
    if not config.use_ica:
        return
    parallel, run_func, _ = parallel_func(apply_ica, n_jobs=config.N_JOBS)
    parallel(run_func(subject, run, session) for subject, run, session in
             itertools.product(config.subjects_list, config.runs,
                               config.sessions))


if __name__ == '__main__':
    main()
