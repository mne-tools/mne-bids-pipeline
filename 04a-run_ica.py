"""
===========
05. Run ICA
===========
This fits ICA on epoched data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.
To actually remove designated ICA components from your data, you will have to
run 06a-apply_ica.py.
"""

import os.path as op
import itertools
import logging

import mne
from mne.report import Report
from mne.preprocessing import ICA
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
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
                                       space=config.space)

    raw_list = list()
    msg = 'Loading filtered raw data'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))

    for run in config.get_runs():
        raw_fname_in = op.join(deriv_path,
                               bids_basename.update(run=run,
                                                    suffix='filt_raw.fif'))
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
        raw_list.append(raw)

    msg = 'Concatenating runs'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))
    raw = mne.concatenate_raws(raw_list)

    events, event_id = mne.events_from_annotations(raw)

    if config.get_kind() == 'eeg':
        raw.set_eeg_reference(projection=True)
    del raw_list

    # don't reject based on EOG to keep blink artifacts
    # in the ICA computation.
    reject_ica = config.get_reject()
    if reject_ica and 'eog' in reject_ica:
        reject_ica = dict(reject_ica)
        del reject_ica['eog']

    # produce high-pass filtered version of the data for ICA
    raw_ica = raw.copy().filter(l_freq=1., h_freq=None)
    epochs_for_ica = mne.Epochs(raw_ica,
                                events, event_id, config.tmin,
                                config.tmax, proj=True,
                                baseline=config.baseline,
                                preload=True, decim=config.decim,
                                reject=reject_ica)

    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(epochs_for_ica.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(epochs_for_ica.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

    # get number of components for ICA
    # compute_rank requires 0.18
    # n_components_meg = (mne.compute_rank(epochs_for_ica.copy()
    #                        .pick_types(meg=True)))['meg']

    n_components_meg = 0.999

    n_components = {'meg': n_components_meg, 'eeg': 0.999}

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

    ica = ICA(method=config.ica_algorithm, random_state=config.random_state,
              n_components=n_components[kind], fit_params=fit_params,
              max_iter=config.ica_max_iterations)

    picks = all_picks[kind]
    if picks.size == 0:
        ica.fit(epochs_for_ica, decim=config.ica_decim)
    else:
        ica.fit(epochs_for_ica, picks=picks, decim=config.ica_decim)

    msg = (f'Fit {ica.n_components_} components (explaining at least '
           f'{100*n_components[kind]:.1f}% of the variance)')
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))

    # Save ICA
    ica_fname = op.join(deriv_path,
                        bids_basename.update(run=None,
                                             suffix=f'{kind}-ica.fif'))
    ica.save(ica_fname)

    if config.interactive:
        # plot ICA components to html report
        report_fname = op.join(deriv_path,
                               bids_basename.update(run=None,
                                                    suffix=f'{kind}-ica.html'))
        report = Report(report_fname, verbose=False)

        for idx in range(0, ica.n_components_):
            figure = ica.plot_properties(epochs_for_ica,
                                         picks=idx,
                                         psd_args={'fmax': 60},
                                         show=False)

            report.add_figs_to_section(figure, section=subject,
                                       captions=(kind.upper() +
                                                 ' - ICA Components'))

        report.save(report_fname, overwrite=True, open_browser=False)


def main():
    """Run ICA."""
    if not config.use_ica:
        return

    msg = 'Running Step 4: Compute ICA'
    logger.info(gen_log_message(step=4, message=msg))

    parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 4: Compute ICA'
    logger.info(gen_log_message(step=4, message=msg))


if __name__ == '__main__':
    main()
