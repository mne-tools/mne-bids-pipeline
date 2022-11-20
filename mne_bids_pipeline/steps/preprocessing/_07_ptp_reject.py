"""Remove epochs based on peak-to-peak (PTP) amplitudes.

Epochs containing peak-to-peak above the thresholds defined
in the 'reject' parameter are removed from the data.

This step will drop epochs containing non-biological artifacts
but also epochs containing biological artifacts not sufficiently
corrected by the ICA or the SSP processing.
"""

from types import SimpleNamespace
from typing import Optional

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import failsafe_run, _update_for_splits, save_logs


def get_input_fnames_drop_ptp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
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
    in_files = dict()
    in_files['epochs'] = bids_path.copy().update(
        processing=cfg.spatial_filter)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_drop_ptp,
)
def drop_ptp(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    out_files = dict()
    out_files['epochs'] = in_files['epochs'].copy().update(processing='clean')
    msg = f'Input:  {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg))
    msg = f'Output: {out_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg))

    # Get rejection parameters and drop bad epochs
    epochs = mne.read_epochs(in_files.pop('epochs'), preload=True)
    reject = _get_reject(
        subject=subject,
        session=session,
        reject=cfg.reject,
        ch_types=cfg.ch_types,
        param='reject',
        epochs=epochs,
    )
    if cfg.spatial_filter == 'ica':
        ica_reject = _get_reject(
            subject=subject,
            session=session,
            reject=cfg.ica_reject,
            ch_types=cfg.ch_types,
            param='ica_reject',
        )
    else:
        ica_reject = None

    if ica_reject is not None:
        for ch_type, threshold in ica_reject.items():
            if (ch_type in reject and
                    threshold < reject[ch_type]):
                # This can only ever happen in case of
                # reject = 'autoreject_global'
                msg = (f'Adjusting PTP rejection threshold proposed by '
                       f'autoreject, as it is greater than ica_reject: '
                       f'{ch_type}: {reject[ch_type]} -> {threshold}')
                logger.info(**gen_log_kwargs(message=msg))
                reject[ch_type] = threshold

    msg = f'Using PTP rejection thresholds: {reject}'
    logger.info(**gen_log_kwargs(message=msg))

    n_epochs_before_reject = len(epochs)
    epochs.reject_tmin = cfg.reject_tmin
    epochs.reject_tmax = cfg.reject_tmax
    epochs.drop_bad(reject=reject)
    n_epochs_after_reject = len(epochs)

    if 0 < n_epochs_after_reject < 0.5 * n_epochs_before_reject:
        msg = ('More than 50% of all epochs rejected. Please check the '
               'rejection thresholds.')
        logger.warning(**gen_log_kwargs(message=msg))
    elif n_epochs_after_reject == 0:
        raise RuntimeError('No epochs remaining after peak-to-peak-based '
                           'rejection. Cannot continue.')

    msg = 'Saving cleaned, baseline-corrected epochs …'

    epochs.apply_baseline(cfg.baseline)
    epochs.save(
        out_files['epochs'], overwrite=True, split_naming='bids',
        split_size=cfg._epochs_split_size)
    _update_for_splits(out_files, 'epochs')
    assert len(in_files) == 0, in_files.keys()

    # Report
    msg = 'Adding cleaned epochs to report.'
    logger.info(**gen_log_kwargs(message=msg))
    # Add PSD plots for 30s of data or all epochs if we have less available
    if len(epochs) * (epochs.tmax - epochs.tmin) < 30:
        psd = True
    else:
        psd = 30
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        report.add_epochs(
            epochs=epochs,
            title='Epochs: after cleaning',
            psd=psd,
            drop_log_ignore=(),
            replace=True,
        )
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        baseline=config.baseline,
        reject_tmin=config.reject_tmin,
        reject_tmax=config.reject_tmax,
        spatial_filter=config.spatial_filter,
        ica_reject=config.ica_reject,
        deriv_root=config.deriv_root,
        reject=config.reject,
        ch_types=config.ch_types,
        _epochs_split_size=config._epochs_split_size,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run epochs."""
    parallel, run_func = parallel_func(
        drop_ptp, exec_params=config.exec_params)

    with get_parallel_backend(config.exec_params):
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session)
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
