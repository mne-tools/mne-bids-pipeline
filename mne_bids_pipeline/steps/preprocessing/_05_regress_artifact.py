"""Temporal regression for artifact removal."""

from types import SimpleNamespace
from typing import Optional

import mne
from mne.io.pick import _picks_to_idx
from mne.preprocessing import EOGRegression

from ..._config_utils import (
    get_runs_tasks,
    get_sessions,
    get_subjects,
)
from ..._import_data import _get_run_rest_noise_path, _get_run_type, _import_data_kwargs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _add_raw, _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def get_input_fnames_regress_artifact(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: str,
    task: Optional[str],
) -> dict:
    """Get paths of files required by regress_artifact function."""
    out = _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="filt",
        mf_reference_run=cfg.mf_reference_run,
    )
    assert len(out)
    return out


@failsafe_run(
    get_input_fnames=get_input_fnames_regress_artifact,
)
def run_regress_artifact(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: str,
    task: Optional[str],
    in_files: dict,
) -> dict:
    model = EOGRegression(proj=False, **cfg.regress_artifact)
    out_files = dict()
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    out_files[in_key] = bids_path_in.copy().update(processing="regress")
    run_type = _get_run_type(run=run, task=task)
    msg = f"Reading {run_type} recording: " f"{bids_path_in.basename}"
    logger.info(**gen_log_kwargs(message=msg))
    raw = mne.io.read_raw_fif(bids_path_in).load_data()
    projs = raw.info["projs"]
    raw.del_proj()
    model.fit(raw)
    all_types = raw.get_channel_types()
    picks = _picks_to_idx(raw.info, model.picks, none="data", exclude=model.exclude)
    ch_types = set(all_types[pick] for pick in picks)
    del picks
    out_files["regress"] = bids_path_in.copy().update(
        processing=None,
        split=None,
        suffix="regress",
        extension=".h5",
    )
    model.apply(raw, copy=False)
    if projs:
        raw.add_proj(projs)
    raw.save(out_files[in_key], overwrite=True, split_size=cfg._raw_split_size)
    _update_for_splits(out_files, in_key)
    model.save(out_files["regress"], overwrite=True)
    assert len(in_files) == 0, in_files.keys()

    # Report
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Adding regressed raw data to report"
        logger.info(**gen_log_kwargs(message=msg))
        figs, captions = list(), list()
        for kind in ("mag", "grad", "eeg"):
            if kind not in ch_types:
                continue
            figs.append(model.plot(ch_type=kind))
            captions.append(f"Run {run}: {kind}")
        if figs:
            report.add_figure(
                fig=figs,
                caption=captions,
                title="Regression weights",
                tags=("raw", f"run-{run}", "regression"),
                replace=True,
            )
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files[in_key],
            title="Raw (regression)",
            tags=("regression",),
            raw=raw,
        )
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        regress_artifact=config.regress_artifact,
        **_import_data_kwargs(config=config, subject=subject),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run artifact regression."""
    if config.regress_artifact is None:
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_regress_artifact, exec_params=config.exec_params
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
