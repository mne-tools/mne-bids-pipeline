"""Apply SSP.

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.
"""

from types import SimpleNamespace

import mne

from mne_bids_pipeline._config_utils import (
    _get_ssrt,
    _get_sst,
    _limit_which_clean,
    _proj_path,
)
from mne_bids_pipeline._import_data import _get_run_rest_noise_path, _import_data_kwargs
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _add_raw, _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_apply_ssp_epochs(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> InFilesT:
    in_files = dict()
    in_files["proj"] = _proj_path(cfg=cfg, subject=subject, session=session)
    in_files["epochs"] = (
        in_files["proj"].copy().update(suffix="epo", check=False, task=task)
    )
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_ssp_epochs,
)
def apply_ssp_epochs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    out_files = dict()
    out_files["epochs"] = (
        in_files["epochs"].copy().update(processing="ssp", split=None, check=False)
    )
    msg = f"Input epochs: {in_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Input SSP:    {in_files['proj'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output:       {out_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)
    projs = mne.read_proj(in_files.pop("proj"))
    epochs_cleaned = epochs.copy().add_proj(projs).apply_proj()

    msg = f"Saving {len(epochs_cleaned)} reconstructed epochs after SSP."
    logger.info(**gen_log_kwargs(message=msg))

    epochs_cleaned.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")
    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_input_fnames_apply_ssp_raw(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> InFilesT:
    in_files = _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="filt",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
    )
    assert len(in_files)
    in_files["proj"] = _proj_path(cfg=cfg, subject=subject, session=session)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_ssp_raw,
)
def apply_ssp_raw(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    projs = mne.read_proj(in_files.pop("proj"))
    in_key = list(in_files.keys())[0]
    assert in_key.startswith("raw"), in_key
    raw_fname = in_files.pop(in_key)
    assert len(in_files) == 0, in_files.keys()
    raw = mne.io.read_raw_fif(raw_fname)
    raw.add_proj(projs)
    out_files = dict()
    out_files[in_key] = raw_fname.copy().update(processing="clean", split=None)
    msg = f"Writing {out_files[in_key].basename} …"
    logger.info(**gen_log_kwargs(message=msg))
    raw.save(out_files[in_key], overwrite=True, split_size=cfg._raw_split_size)
    _update_for_splits(out_files, in_key)
    # Report
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Adding cleaned raw data to report"
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files[in_key],
            title_prefix="Raw (clean)",
            tags=("clean",),
            raw=raw,
        )
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        processing="filt" if config.regress_artifact is None else "regress",
        _epochs_split_size=config._epochs_split_size,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Apply ssp."""
    if not config.spatial_filter == "ssp":
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    sst = _get_sst(config=config)
    which = _limit_which_clean(config=config)
    ssrt = _get_ssrt(config=config, which=which)
    with get_parallel_backend(config.exec_params):
        # Epochs
        parallel, run_func = parallel_func(
            apply_ssp_epochs, exec_params=config.exec_params, n_iter=len(sst)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                task=task,
            )
            for subject, session, task in sst
        )
        # Raw
        parallel, run_func = parallel_func(
            apply_ssp_raw, exec_params=config.exec_params, n_iter=len(ssrt)
        )
        logs += parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )
    save_logs(config=config, logs=logs)
