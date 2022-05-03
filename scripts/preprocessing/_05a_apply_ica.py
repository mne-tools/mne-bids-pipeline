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
from typing import Optional
from types import SimpleNamespace

import pandas as pd
import mne
from mne.preprocessing import read_ica
from mne.report import Report

from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def apply_ica(*, cfg, subject, session):
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

    report_fname = (bids_basename.copy()
                    .update(processing='ica', suffix='report',
                            extension='.html'))

    title = f'ICA artifact removal – sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    # Load ICA.
    msg = f'Reading ICA: {fname_ica}'
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))
    ica = read_ica(fname=fname_ica)

    # Select ICs to remove.
    tsv_data = pd.read_csv(fname_ica_components, sep='\t')
    ica.exclude = (tsv_data
                   .loc[tsv_data['status'] == 'bad', 'component']
                   .to_list())

    # Load epochs to reject ICA components.
    msg = f'Input: {fname_epo_in}, Output: {fname_epo_out}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(fname_epo_in, preload=True)
    epochs.drop_bad(cfg.ica_reject)

    # Now actually reject the components.
    msg = f'Rejecting ICs: {", ".join([str(ic) for ic in ica.exclude])}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

    msg = 'Saving reconstructed epochs after ICA.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_cleaned.save(fname_epo_out, overwrite=True, split_naming='bids')

    # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # response is calculated across ALL epochs, just like ICA was run on
    # all epochs, regardless of their respective experimental condition.
    #
    # We apply baseline correction here to (hopefully!) make the effects of
    # ICA easier to see. Otherwise, individual channels might just have
    # arbitrary DC shifts, and we wouldn't be able to easily decipher what's
    # going on!
    report = Report(report_fname, title=title, verbose=False)
    picks = ica.exclude if ica.exclude else None
    report.add_ica(
        ica=ica,
        title='Effects of ICA cleaning',
        inst=epochs.copy().apply_baseline(cfg.baseline),
        picks=picks
    )
    report.save(report_fname, overwrite=True, open_browser=cfg.interactive)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        interactive=config.interactive,
        baseline=config.baseline,
        ica_reject=config.get_ica_reject()
    )
    return cfg


def main():
    """Apply ICA."""
    if not config.spatial_filter == 'ica':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(apply_ica)
        logs = parallel(
            run_func(cfg=get_config(), subject=subject, session=session)
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
