"""Apply ICA.

!! If you manually add components to remove, make sure you did not re-run the ICA in
the meantime. Otherwise (especially if the random state was not set, or you used a
different machine) the component order might differ.
"""

from types import SimpleNamespace

import mne
import pandas as pd
from mne.preprocessing import read_ica
from mne_bids import BIDSPath

from ..._config_utils import (
    get_runs_tasks,
    get_sessions,
    get_subjects,
)
from ..._import_data import _get_run_rest_noise_path, _import_data_kwargs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _add_raw, _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def _ica_paths(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
):
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files = dict()
    in_files["ica"] = bids_basename.copy().update(
        processing="ica",
        suffix="ica",
        extension=".fif",
    )
    in_files["components"] = bids_basename.copy().update(
        processing="ica", suffix="components", extension=".tsv"
    )
    return in_files


def _read_ica_and_exclude(
    in_files: dict,
) -> None:
    ica = read_ica(fname=in_files.pop("ica"))
    tsv_data = pd.read_csv(in_files.pop("components"), sep="\t")
    ica.exclude = tsv_data.loc[tsv_data["status"] == "bad", "component"].to_list()
    return ica


def get_input_fnames_apply_ica_epochs(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> dict:
    in_files = _ica_paths(cfg=cfg, subject=subject, session=session)
    in_files["epochs"] = (
        in_files["ica"]
        .copy()
        .update(
            suffix="epo",
            extension=".fif",
            processing=None,
        )
    )
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


def get_input_fnames_apply_ica_raw(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> dict:
    in_files = _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="filt",
        mf_reference_run=cfg.mf_reference_run,
    )
    assert len(in_files)
    in_files.update(_ica_paths(cfg=cfg, subject=subject, session=session))
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_ica_epochs,
)
def apply_ica_epochs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: dict,
) -> dict:
    out_files = dict()
    out_files["epochs"] = in_files["epochs"].copy().update(processing="ica", split=None)

    title = f"ICA artifact removal – sub-{subject}"
    if session is not None:
        title += f", ses-{session}"
    if cfg.task is not None:
        title += f", task-{cfg.task}"

    # Load ICA.
    msg = f"Reading ICA: {in_files['ica']}"
    logger.debug(**gen_log_kwargs(message=msg))
    ica = _read_ica_and_exclude(in_files)

    # Load epochs.
    msg = f'Input: {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg))
    msg = f'Output: {out_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg))

    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    # Now actually reject the components.
    msg = (
        f'Rejecting ICs with the following indices: '
        f'{", ".join([str(i) for i in ica.exclude])}'
    )
    logger.info(**gen_log_kwargs(message=msg))
    epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

    msg = f"Saving {len(epochs)} reconstructed epochs after ICA."
    logger.info(**gen_log_kwargs(message=msg))
    epochs_cleaned.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")
    assert len(in_files) == 0, in_files.keys()

    # Report
    kwargs = dict()
    if ica.exclude:
        msg = "Adding ICA to report."
    else:
        msg = "Skipping ICA addition to report, no components marked as bad."
        kwargs["emoji"] = "skip"
    logger.info(**gen_log_kwargs(message=msg, **kwargs))
    if ica.exclude:
        with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session,
        ) as report:
            report.add_ica(
                ica=ica,
                title="ICA: removals",
                inst=epochs,
                picks=ica.exclude,
                # TODO upstream
                # captions=f'Evoked response (across all epochs) '
                # f'before and after ICA '
                # f'({len(ica.exclude)} ICs removed)'
                replace=True,
                n_jobs=1,  # avoid automatic parallelization
            )

    return _prep_out_files(exec_params=exec_params, out_files=out_files)


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_ica_raw,
)
def apply_ica_raw(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    in_files: dict,
) -> dict:
    ica = _read_ica_and_exclude(in_files)
    in_key = list(in_files)[0]
    assert in_key.startswith("raw"), in_key
    raw_fname = in_files.pop(in_key)
    assert len(in_files) == 0, in_files
    out_files = dict()
    out_files[in_key] = raw_fname.copy().update(processing="clean", split=None)
    msg = f"Writing {out_files[in_key].basename} …"
    logger.info(**gen_log_kwargs(message=msg))
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    ica.apply(raw)
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
            title="Raw (clean)",
            tags=("clean",),
            raw=raw,
        )
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        baseline=config.baseline,
        ica_reject=config.ica_reject,
        processing="filt" if config.regress_artifact is None else "regress",
        _epochs_split_size=config._epochs_split_size,
        **_import_data_kwargs(config=config, subject=subject),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Apply ICA."""
    if not config.spatial_filter == "ica":
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        # Epochs
        parallel, run_func = parallel_func(
            apply_ica_epochs, exec_params=config.exec_params
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
        # Raw
        parallel, run_func = parallel_func(
            apply_ica_raw, exec_params=config.exec_params
        )
        logs += parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for run, task in get_runs_tasks(
                config=config,
                subject=subject,
                session=session,
            )
        )
    save_logs(config=config, logs=logs)
