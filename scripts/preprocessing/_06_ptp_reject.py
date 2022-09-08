"""
========================================================
06. Remove epochs based on peak-to-peak (PTP) amplitudes
========================================================

Epochs containing peak-to-peak above the thresholds defined
in the 'reject' parameter are removed from the data.

This step will drop epochs containing non-biological artifacts
but also epochs containing biological artifacts not sufficiently
corrected by the ICA or the SSP processing.
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def drop_ptp(*, cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix='epo',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    infile_processing = cfg.spatial_filter
    fname_in = bids_path.copy().update(processing=infile_processing)
    fname_out = bids_path.copy().update(processing='clean')

    msg = f'Input: {fname_in.basename}, Output: {fname_out.basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    # Get rejection parameters and drop bad epochs
    epochs = mne.read_epochs(fname_in, preload=True)
    reject = config.get_reject(epochs=epochs)

    if cfg.ica_reject is not None:
        for ch_type, threshold in cfg.ica_reject.items():
            if (ch_type in reject and
                    threshold < reject[ch_type]):
                # This can only ever happen in case of
                # reject = 'autoreject_global'
                msg = (f'Adjusting PTP rejection threshold proposed by '
                       f'autoreject, as it is greater than ica_reject: '
                       f'{ch_type}: {reject[ch_type]} -> {threshold}')
                logger.info(**gen_log_kwargs(message=msg,
                                             subject=subject, session=session))
                reject[ch_type] = threshold

    msg = f'Using PTP rejection thresholds: {reject}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    n_epochs_before_reject = len(epochs)
    epochs.reject_tmin = cfg.reject_tmin
    epochs.reject_tmax = cfg.reject_tmax
    epochs.drop_bad(reject=reject)
    n_epochs_after_reject = len(epochs)

    if 0 < n_epochs_after_reject < 0.5 * n_epochs_before_reject:
        msg = ('More than 50% of all epochs rejected. Please check the '
               'rejection thresholds.')
        logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                        session=session))
    elif n_epochs_after_reject == 0:
        raise RuntimeError('No epochs remaining after peak-to-peak-based '
                           'rejection. Cannot continue.')

    msg = 'Saving cleaned, baseline-corrected epochs …'

    epochs.apply_baseline(cfg.baseline)
    epochs.save(
        fname_out, overwrite=True, split_naming='bids',
        split_size=cfg._epochs_split_size)


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
        baseline=config.baseline,
        reject_tmin=config.reject_tmin,
        reject_tmax=config.reject_tmax,
        spatial_filter=config.spatial_filter,
        ica_reject=config.get_ica_reject(),
        deriv_root=config.get_deriv_root(),
        decim=config.decim,
        _epochs_split_size=config._epochs_split_size,
    )
    return cfg


def main():
    """Run epochs."""
    parallel, run_func = parallel_func(drop_ptp)

    with config.get_parallel_backend():
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
