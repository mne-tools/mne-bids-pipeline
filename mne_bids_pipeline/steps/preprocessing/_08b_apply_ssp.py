"""Apply SSP projections and obtain the cleaned epochs and raw data.

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

from types import SimpleNamespace
from typing import Optional

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    _bids_kwargs,
    get_sessions,
    get_subjects,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def get_input_fnames_apply_ssp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        extension=".fif",
        check=False,
    )
    in_files = dict()
    in_files["epochs"] = bids_basename.copy().update(suffix="epo", check=False)
    _update_for_splits(in_files, "epochs", single=True)
    in_files["proj"] = bids_basename.copy().update(suffix="proj", check=False)
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
    out_files = dict()
    out_files["epochs"] = (
        in_files["epochs"].copy().update(processing="ssp", split=None, check=False)
    )
    msg = f"Input epochs: {in_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f'Input SSP:    {in_files["proj"].basename}'
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output:       {out_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)
    projs = mne.read_proj(in_files.pop("proj"))
    epochs_cleaned = epochs.copy().add_proj(projs).apply_proj()
    epochs_cleaned.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")
    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        _epochs_split_size=config._epochs_split_size,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Apply ssp."""
    if not config.spatial_filter == "ssp":
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(apply_ssp, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
