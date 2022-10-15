"""Remove epochs based on peak-to-peak (PTP) amplitudes.

Epochs containing peak-to-peak above the thresholds defined
in the 'reject' parameter are removed from the data.

This step will drop epochs containing non-biological artifacts
but also epochs containing biological artifacts not sufficiently
corrected by the ICA or the SSP processing.
"""

import itertools
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype,
    get_deriv_root,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run, _update_for_splits, save_logs
from ..._reject import _get_reject


def get_input_fnames_drop_ptp(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
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


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_drop_ptp)
def drop_ptp(*, cfg, subject, session, in_files):
    out_files = dict()
    out_files['epochs'] = in_files['epochs'].copy().update(processing='clean')
    msg = f'Input:  {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f'Output: {out_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

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
        out_files['epochs'], overwrite=True, split_naming='bids',
        split_size=cfg._epochs_split_size)
    _update_for_splits(out_files, 'epochs')
    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config,
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
        deriv_root=get_deriv_root(config),
        reject=config.reject,
        ch_types=config.ch_types,
        _epochs_split_size=config._epochs_split_size,
    )
    return cfg


def main():
    """Run epochs."""
    import config
    parallel, run_func = parallel_func(drop_ptp, config=config)

    with get_parallel_backend(config):
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                subject=subject,
                session=session)
            for subject, session in
            itertools.product(
                get_subjects(config),
                get_sessions(config)
            )
        )
    save_logs(config=config, logs=logs)


if __name__ == '__main__':
    main()
