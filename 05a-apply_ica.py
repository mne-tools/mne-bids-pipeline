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

import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.report import Report

from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def apply_ica(subject, session):
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
                                       space=config.space,
                                       prefix=deriv_path)

    fname_in = bids_basename.copy().update(suffix='epo.fif')
    fname_out = bids_basename.copy().update(suffix='cleaned_epo.fif')

    # load epochs to reject ICA components
    epochs = mne.read_epochs(fname_in, preload=True)

    msg = f'Input: {fname_in}, Output: {fname_out}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    # load first run of raw data for ecg / eog epochs
    msg = 'Loading first run from raw data'
    logger.debug(gen_log_message(message=msg, step=5, subject=subject,
                                 session=session))

    if config.use_maxwell_filter:
        raw_fname_in = (bids_basename.copy()
                        .update(run=config.get_runs()[0],
                                suffix='sss_raw.fif'))
    else:
        raw_fname_in = (bids_basename.copy()
                        .update(run=config.get_runs()[0],
                                suffix='filt_raw.fif'))

    raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

    for ch_type in config.ch_types:
        report_fname = (bids_basename.copy()
                        .update(suffix=f'ica-reject.html'))
        report = Report(report_fname, verbose=False)

        # Load ICA
        fname_ica = (bids_basename.copy()
                     .update(run=None, suffix=f'ica.fif'))

        msg = f'Reading ICA: {fname_ica}'
        logger.debug(gen_log_message(message=msg, step=5, subject=subject,
                                     session=session))
        ica = read_ica(fname=fname_ica)

        # ECG either needs an ecg channel, or avg of the mags (i.e. MEG data)
        if 'ecg' in raw.get_channel_types() or ch_type in ('meg', 'mag'):
            # Create ecg epochs
            if ch_type == 'meg':
                reject = {'mag': config.reject['mag'],
                          'grad': config.reject['grad']}
            elif ch_type == 'mag':
                reject = {'mag': config.reject['mag']}
            elif ch_type == 'eeg':
                reject = {'eeg': config.reject['eeg']}

            ecg_epochs = create_ecg_epochs(raw, reject=reject,
                                           baseline=(None, 0),
                                           tmin=-0.5, tmax=0.5)

            ecg_average = ecg_epochs.average()

            ecg_inds, scores = \
                ica.find_bads_ecg(ecg_epochs, method='ctps',
                                  threshold=config.ica_ctps_ecg_threshold)
            ica.exclude = ecg_inds
            del ecg_epochs

            # Plot r score
            report.add_figs_to_section(
                ica.plot_scores(scores, show=config.interactive),
                captions='ECG - R scores')

            # Plot source time course
            report.add_figs_to_section(
                ica.plot_sources(ecg_average, show=config.interactive),
                captions='ECG - Sources time course')

            # Plot source time course
            report.add_figs_to_section(
                ica.plot_overlay(ecg_average, show=config.interactive),
                captions='ECG - Corrections')

        else:
            # XXX : to check when EEG only is processed
            ecg_inds = list()
            msg = ('No ECG channel is present. Cannot automate IC detection '
                   'for ECG')
            logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                        session=session))

        # EOG
        pick_eog = mne.pick_types(raw.info, meg=False, eeg=False,
                                  ecg=False, eog=True)
        if pick_eog.any():
            msg = 'Using EOG channel'
            logger.debug(gen_log_message(message=msg, step=5, subject=subject,
                                         session=session))

            # Create eog epochs
            eog_epochs = create_eog_epochs(raw, reject=None,
                                           baseline=(None, 0),
                                           tmin=-0.5, tmax=0.5)

            eog_average = eog_epochs.average()
            eog_inds, scores = ica.find_bads_eog(eog_epochs, threshold=3.0)
            ica.exclude = eog_inds
            del eog_epochs

            # Plot r score
            report.add_figs_to_section(
                ica.plot_scores(scores, show=config.interactive),
                captions='EOG - R scores')

            # Plot source time course
            report.add_figs_to_section(
                ica.plot_sources(eog_average, show=config.interactive),
                captions=f'EOG - Sources time course')

            # Plot source time course
            report.add_figs_to_section(
                ica.plot_overlay(eog_average, show=config.interactive),
                captions='EOG - Corrections')
        else:
            eog_inds = list()
            msg = ('No EOG channel is present. Cannot automate IC detection '
                   'for EOG')
            logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                        session=session))

        # Now compile a list of all ICs that should be rejected, and add them
        # to the ICA object. From then on, the rejected ICs will be handled
        # correctly automatically (e.g. no need to pass `exclude=...` to the
        # ICA plotting functions etc.).
        exclude_ics = (list(ecg_inds) + list(eog_inds) +
                       list(config.rejcomps_man[subject][ch_type]))
        ica.exclude = exclude_ics

        # Compare ERP/ERF before and after ICA artifact rejection. The evoked
        # response is calculated across ALL epochs, just like ICA was run on
        # all epochs, regardless of their respective experimental condition.
        #
        # Note that until now, we haven't actually rejected any ICs from the
        # epochs.
        fig = ica.plot_overlay(epochs.average(),
                               show=config.interactive)
        report.add_figs_to_section(
            fig,
            captions='Evoked response (across all epochs) before and after '
                     'ICA')
        report.save(report_fname, overwrite=True, open_browser=False)

        # Now actually reject the components.
        msg = f'Rejecting from {ch_type}: {ica.exclude}'
        logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                    session=session))
        epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

        msg = ('Saving cleaned epochs and updated ICA object (with list of '
               'excluded components)')
        logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                    session=session))
        epochs_cleaned.save(fname_out, overwrite=True)
        ica.save(fname_ica)

        if config.interactive:
            epochs_cleaned.plot_image(combine='gfp', group_by='type', sigma=2.,
                                      cmap="YlGnBu_r")


def main():
    """Apply ICA."""
    if not config.use_ica:
        return

    msg = 'Running Step 5: Apply ICA'
    logger.info(gen_log_message(step=5, message=msg))

    parallel, run_func, _ = parallel_func(apply_ica, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 5: Apply ICA'
    logger.info(gen_log_message(step=5, message=msg))


if __name__ == '__main__':
    main()
