"""
===========
05. Run ICA
===========
This fits ICA on epoched data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.

To actually remove designated ICA components from your data, you will have to
run 05a-apply_ica.py.
"""

import itertools
import logging
import os.path as op

import mne
from mne.report import Report
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def load_and_concatenate_raws(bids_basename):
    raw_list = list()

    for run in config.get_runs():
        raw_fname_in = (bids_basename.copy()
                        .update(run=run, suffix='filt_raw.fif'))
        raw = mne.io.read_raw_fif(raw_fname_in, preload=False)
        raw_list.append(raw)

    raw = mne.concatenate_raws(raw_list)
    del raw_list

    if config.get_kind() == 'eeg':
        raw.set_eeg_reference(projection=True)

    raw.load_data()
    return raw


def filter_for_ica(raw, subject, session):
    """Apply a high-pass filter if needed."""
    if config.ica_l_freq == config.l_freq or config.ica_l_freq is None:
        # Nothing to do here!
        msg = 'Not applying high-pass filter.'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
    else:
        msg = f'Applying high-pass filter with {config.ica_l_freq} Hz cutoff …'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
        raw.filter(l_freq=config.ica_l_freq, h_freq=None)

    return raw


def make_epochs_for_ica(raw):
    """Epoch the raw data."""

    # don't reject epochs based on EOG to keep blink artifacts
    # in the ICA computation.
    reject_ica = config.get_reject()
    if reject_ica and 'eog' in reject_ica:
        reject_ica = reject_ica.copy()
        del reject_ica['eog']

    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw,
                        events, event_id, config.tmin,
                        config.tmax, proj=True,
                        baseline=config.baseline,
                        preload=True, decim=config.decim,
                        reject=reject_ica)
    return epochs


def fit_ica(epochs, subject, session):
    kind = config.get_kind()
    msg = f'Running ICA for {kind}'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))

    if config.ica_algorithm == 'picard':
        fit_params = dict(fastica_it=5)
    elif config.ica_algorithm == 'extended_infomax':
        fit_params = dict(extended=True)
    elif config.ica_algorithm == 'fastica':
        fit_params = None

    # XXX get number of components for ICA
    # compute_rank requires 0.18
    # n_components_meg = (mne.compute_rank(epochs_for_ica.copy()
    #                        .pick_types(meg=True)))['meg']

    ica = ICA(method=config.ica_algorithm, random_state=config.random_state,
              n_components=config.ica_n_components, fit_params=fit_params,
              max_iter=config.ica_max_iterations)

    ica.fit(epochs, decim=config.ica_decim)

    explained_var = (ica.pca_explained_variance_[:ica.n_components_].sum() /
                     ica.pca_explained_variance_.sum())
    msg = (f'Fit {ica.n_components_} components (explaining '
           f'{round(explained_var * 100, 1)}% of the variance) in {ica.n_iter_} '
           f'iterations.')
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))
    return ica


def detect_ecg_artifacts(ica, raw, subject, session, report):
    # ECG either needs an ecg channel, or avg of the mags (i.e. MEG data)
    if ('ecg' in raw.get_channel_types() or 'meg' in config.ch_types or
            'mag' in config.ch_types):
        msg = 'Performing automated ECG artifact detection …'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

        # Create ecg epochs
        # Don't reject epochs based on ECG to retain artifacts.
        reject = config.get_reject()
        if reject and 'ecg' in reject:
            reject = reject.copy()
            del reject['ecg']

        ecg_epochs = create_ecg_epochs(raw, reject=reject,
                                       baseline=(None, -0.2),
                                       tmin=-0.5, tmax=0.5)
        ecg_average = ecg_epochs.average()

        ecg_inds, scores = ica.find_bads_ecg(
            ecg_epochs, method='ctps',
            threshold=config.ica_ctps_ecg_threshold)
        ica.exclude = ecg_inds

        msg = (f'Detected {len(ecg_inds)} ECG-related ICs in '
               f'{len(ecg_epochs)} ECG epochs.')
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
        del ecg_epochs

        # Plot scores
        fig = ica.plot_scores(scores, show=config.interactive)
        report.add_figs_to_section(figs=fig, captions='Scores - ECG',
                                   section=f'sub-{subject}')

        # Plot source time course
        fig = ica.plot_sources(ecg_average, show=config.interactive)
        report.add_figs_to_section(figs=fig,
                                   captions='Source time course - ECG',
                                   section=f'sub-{subject}')

        # Plot original & corrected data
        fig = ica.plot_overlay(ecg_average, show=config.interactive)
        report.add_figs_to_section(figs=fig, captions='Corrections - ECG',
                                   section=f'sub-{subject}')
    else:
        ecg_inds = list()
        msg = ('No ECG or magnetometer channels are present. Cannot '
               'automate artifact detection for ECG')
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

    return ecg_inds


def detect_eog_artifacts(ica, raw, subject, session, report):
    # EOG
    pick_eog = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False,
                              eog=True)
    if pick_eog.any():
        msg = 'Performing automated EOG artifact detection …'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

        # Create EOG epochs.
        # Don't reject epochs based on EOG to retain artifacts.
        eog_epochs = create_eog_epochs(raw, reject=None,
                                       baseline=(None, -0.2),
                                       tmin=-0.5, tmax=0.5)

        eog_average = eog_epochs.average()
        eog_inds, scores = ica.find_bads_eog(
            eog_epochs,
            threshold=config.ica_eog_threshold)
        ica.exclude = eog_inds

        msg = (f'Detected {len(eog_inds)} EOG-related ICs in '
               f'{len(eog_epochs)} EOG epochs.')
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
        del eog_epochs

        # Plot scores
        fig = ica.plot_scores(scores, show=config.interactive)
        report.add_figs_to_section(figs=fig, captions='Scores - EOG',
                                   section=f'sub-{subject}')

        # Plot source time course
        fig = ica.plot_sources(eog_average, show=config.interactive)
        report.add_figs_to_section(figs=fig,
                                   captions='Source time course - EOG',
                                   section=f'sub-{subject}')

        # Plot original & corrected data
        fig = ica.plot_overlay(eog_average, show=config.interactive)
        report.add_figs_to_section(figs=fig, captions='Corrections - EOG',
                                   section=f'sub-{subject}')
    else:
        eog_inds = list()
        msg = ('No EOG channel is present. Cannot automate IC detection '
               'for EOG')
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

    return eog_inds


def run_ica(subject, session=None):
    """Run ICA."""

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path)

    ica_fname = bids_basename.copy().update(suffix='ica.fif')
    report_fname = (bids_basename.copy().update(suffix='ica.html'))

    msg = 'Loading and concatenating filtered continuous "raw" data'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))
    raw = load_and_concatenate_raws(bids_basename)

    # Produce high-pass filtered version of the data for ICA.
    # filter_for_ica will concatenate all runs of our raw data.
    # We don't have to worry about edge artifacts due to raw concatenation as
    # we'll be epoching the data in the next step.
    raw = filter_for_ica(raw, subject=subject, session=session)
    epochs = make_epochs_for_ica(raw)

    # Now actually perform ICA, or load from disk if the user specified ICs
    # for rejection in the configuration file -- we want to avoid
    # re-calculation of ICA in that case!
    if isinstance(config.ica_reject_components, dict) and op.exists(ica_fname):
        msg = (f'Loading existing ICA solution from disk, because you '
               f'components for rejection via ica_reject_components in your '
               f'configuration file. If you want to generate a new ICA '
               f'solution, either remove the ICs from ica_reject_components, '
               f'or delete the ICA solution file {ica_fname}')
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
        ica = mne.preprocessing.read_ica(ica_fname)
    else:
        msg = 'Calculating ICA solution.'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))
        ica = fit_ica(epochs, subject=subject, session=session)

    msg = ('Creating HTML report …')
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))

    report = Report(report_fname, title='Independent Component Analysis (ICA)',
                    verbose=False)

    ecg_ics = detect_ecg_artifacts(ica=ica, raw=raw, subject=subject,
                                   session=session, report=report)
    eog_ics = detect_eog_artifacts(ica=ica, raw=raw, subject=subject,
                                   session=session, report=report)

    # Save ICA to disk.
    # We also store the automatically identified ECG- and EOG-related ICs.
    msg = 'Saving ICA solution and detected artifacts to disk.'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))
    ica.exclude = sorted(set(ecg_ics + eog_ics))
    ica.save(ica_fname)

    # Lastly, plot all ICs, and add them to the report for manual inspection.
    msg = ('Adding diagnostic plots for all ICs to the report …')
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))
    for component_num in range(ica.n_components_):
        fig = ica.plot_properties(epochs,
                                  picks=component_num,
                                  psd_args={'fmax': 60},
                                  show=False)

        report.add_figs_to_section(fig, section=f'sub-{subject}',
                                   captions=f'IC {component_num}')

    open_browser = True if config.interactive else False
    report.save(report_fname, overwrite=True, open_browser=open_browser)

    msg = (f"ICA completed. Please carefully review the extracted ICs in the "
           f"report, and add all ICs you wish to exclude to the configuration "
           f"file, e.g.: "
           f"ica_reject_components = {{'sub-{subject}': [0, 1, 5 ]}}")
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))


@failsafe_run(on_error=on_error)
def main():
    """Run ICA."""
    msg = 'Running Step 4: Compute ICA'
    logger.info(gen_log_message(step=4, message=msg))

    if config.use_ica:
        parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
        parallel(run_func(subject, session) for subject, session in
                 itertools.product(config.get_subjects(),
                                   config.get_sessions()))

    msg = 'Completed Step 4: Compute ICA'
    logger.info(gen_log_message(step=4, message=msg))


if __name__ == '__main__':
    main()
