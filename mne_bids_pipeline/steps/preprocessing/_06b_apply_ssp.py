"""Apply SSP projections and obtain the cleaned epochs.

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

from types import SimpleNamespace
from typing import Optional

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype,
)
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, _update_for_splits, save_logs
from ..._parallel import parallel_func, get_parallel_backend


def get_input_fnames_apply_ssp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    bids_basename = BIDSPath(subject=subject,
                             session=session,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             recording=cfg.rec,
                             space=cfg.space,
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             extension='.fif',
                             check=False)
    in_files = dict()
    in_files['epochs'] = bids_basename.copy().update(suffix='epo', check=False)
    in_files['proj'] = bids_basename.copy().update(suffix='proj', check=False)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_ssp,
)
def apply_ssp(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    # load epochs to reject ICA components
    # compute SSP on first run of raw
    out_files = dict()
    out_files['epochs'] = in_files['epochs'].copy().update(
        processing='ssp', check=False)
    msg = f"Input epochs: {in_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f'Input SSP:    {in_files["proj"].basename}'
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output:       {out_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    epochs = mne.read_epochs(in_files.pop('epochs'), preload=True)
    projs = mne.read_proj(in_files.pop('proj'))
    epochs_cleaned = epochs.copy().add_proj(projs).apply_proj()
    epochs_cleaned.save(
        out_files['epochs'], overwrite=True, split_naming='bids',
        split_size=cfg._epochs_split_size)
    _update_for_splits(out_files, 'epochs')
    assert len(in_files) == 0, in_files.keys()
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
        deriv_root=config.deriv_root,
        _epochs_split_size=config._epochs_split_size,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Apply ssp."""
    if not config.spatial_filter == 'ssp':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            apply_ssp, exec_params=config.exec_params)
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
