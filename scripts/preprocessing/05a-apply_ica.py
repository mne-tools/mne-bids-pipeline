"""
===============
05. Apply ICA
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

import pandas as pd

import mne
from mne.utils._bunch import BunchConst
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.report import Report

from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def apply_ica(cfg, subject, session):
    bids_basename = BIDSPath(subject=subject,
                             session=session,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             run=None,
                             recording=cfg.rec,
                             space=cfg.space,
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             check=False)

    fname_epo_in = bids_basename.copy().update(suffix='epo', extension='.fif')
    fname_epo_out = bids_basename.copy().update(
        processing='ica', suffix='epo', extension='.fif')
    fname_ica = bids_basename.copy().update(suffix='ica', extension='.fif')
    fname_ica_components = bids_basename.copy().update(
        processing='ica', suffix='components', extension='.tsv')

    # Load epochs to reject ICA components.
    epochs = mne.read_epochs(fname_epo_in, preload=True)

    msg = f'Input: {fname_epo_in}, Output: {fname_epo_out}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    report_fname = (bids_basename.copy()
                    .update(processing='ica', suffix='report',
                            extension='.html'))

    title = f'ICA artifact removal – sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'
    report = Report(report_fname, title=title, verbose=False)

    # Load ICA.
    msg = f'Reading ICA: {fname_ica}'
    logger.debug(gen_log_message(message=msg, step=5, subject=subject,
                                 session=session))
    ica = read_ica(fname=fname_ica)

    # Select ICs to remove.
    tsv_data = pd.read_csv(fname_ica_components, sep='\t')
    ica.exclude = (tsv_data
                   .loc[tsv_data['status'] == 'bad', 'component']
                   .to_list())

    # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # response is calculated across ALL epochs, just like ICA was run on
    # all epochs, regardless of their respective experimental condition.
    #
    # Note that up until now, we haven't actually rejected any ICs from the
    # epochs.
    #
    # We apply baseline correction here to (hopefully!) make the effects of
    # ICA easier to see. Otherwise, individual channels might just have
    # arbitrary DC shifts, and we wouldn't be able to easily decipher what's
    # going on!
    evoked = epochs.average().apply_baseline(cfg.baseline)

    # Plot source time course
    fig = ica.plot_sources(evoked, show=cfg.interactive)
    report.add_figs_to_section(figs=fig,
                               captions='All ICs - Source time course')

    # Plot original & corrected data
    fig = ica.plot_overlay(evoked, show=cfg.interactive)
    report.add_figs_to_section(figs=fig,
                               captions=f'Evoked response (across all epochs) '
                                        f'before and after cleaning via ICA '
                                        f'({len(ica.exclude)} ICs removed)')
    report.save(report_fname, overwrite=True, open_browser=False)

    # Now actually reject the components.
    msg = f'Rejecting ICs: {", ".join([str(ic) for ic in ica.exclude])}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

    msg = 'Saving reconstructed epochs after ICA.'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned.save(fname_epo_out, overwrite=True)

    if cfg.interactive:
        epochs_cleaned.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")


def get_config():
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        interactive=config.interactive,
        baseline=config.baseline,
        spatial_filter=config.spatial_filter,
        subjects=config.get_subjects(),
        sessions=config.get_sessions(),
        N_JOBS=config.N_JOBS
    )
    return cfg


def main():
    """Apply ICA."""

    if not config.spatial_filter == 'ica':
        return

    msg = 'Running Step 5: Apply ICA'
    logger.info(gen_log_message(step=5, message=msg))

    cfg = get_config()

    parallel, run_func, _ = parallel_func(apply_ica, n_jobs=config.N_JOBS)
    parallel(run_func(cfg, subject, session)
             for subject, session in
             itertools.product(cfg.subjects, cfg.sessions))

    msg = 'Completed Step 5: Apply ICA'
    logger.info(gen_log_message(step=5, message=msg))


if __name__ == '__main__':
    main()
