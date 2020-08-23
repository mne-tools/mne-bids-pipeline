"""
===============
06. Apply ICA
===============

Blinks and ECG artifacts are automatically detected and the corresponding ICA
components are removed from the data.
This relies on the ICAs computed in 04-run_ica.py

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
from mne.report import Report

from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def apply_ica(subject, session):
    kind = config.get_kind()
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path)

    fname_epo_in = bids_basename.copy().update(kind='epo', extension='.fif')
    fname_epo_out = bids_basename.copy().update(kind='epo', processing='clean',
                                                extension='.fif')
    fname_ica = bids_basename.copy().update(kind='ica', extension='.fif')

    # Load epochs to reject ICA components.
    epochs = mne.read_epochs(fname_epo_in, preload=True)

    msg = f'Input: {fname_epo_in}, Output: {fname_epo_out}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    report_fname = (bids_basename.copy()
                    .update(processing='clean', kind='report',
                            extension='.html'))
    report = Report(report_fname, verbose=False)

    # Load ICA.
    msg = f'Reading ICA: {fname_ica}'
    logger.debug(gen_log_message(message=msg, step=5, subject=subject,
                                 session=session))
    ica = read_ica(fname=fname_ica)

    # Select ICs to remove.
    if config.ica_reject_components is None:
        # Keep all ICs. User simply wanted the ICA report, but doesn't wish to
        # reject any ICs.
        ica.exclude = []
    elif config.ica_reject_components == 'auto':
        # We saved the ECG and EOG components to ica.exclude in the previous
        # processing step.
        pass
    else:
        ica.exclude = config.ica_reject_components[subject]

    # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # response is calculated across ALL epochs, just like ICA was run on
    # all epochs, regardless of their respective experimental condition.
    #
    # Note that up until now, we haven't actually rejected any ICs from the
    # epochs.

    evoked = epochs.average()

    # Plot source time course
    fig = ica.plot_sources(evoked, show=config.interactive)
    report.add_figs_to_section(figs=fig,
                               captions='All ICs - Source time course')

    # Plot original & corrected data
    fig = ica.plot_overlay(evoked, show=config.interactive)
    report.add_figs_to_section(figs=fig,
                               captions='Evoked response (across all epochs) '
                                        'before and after IC removal')
    report.save(report_fname, overwrite=True, open_browser=False)

    # Now actually reject the components.
    msg = f'Rejecting ICs: {ica.exclude}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

    msg = 'Saving cleaned epochs.'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned.save(fname_epo_out, overwrite=True)

    if config.interactive:
        epochs_cleaned.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")


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
