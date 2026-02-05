"""Temporal regression for artifact removal."""

from types import SimpleNamespace

import mne
from mne.io.pick import _picks_to_idx
from mne.preprocessing import EOGRegression

from mne_bids_pipeline._config_utils import _get_ssrt
from mne_bids_pipeline._import_data import (
    _get_run_rest_noise_path,
    _import_data_kwargs,
    _read_raw_msg,
)
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


def get_input_fnames_regress_artifact(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by regress_artifact function."""
    out = _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="filt",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
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
    session: str | None,
    run: str,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    model = EOGRegression(proj=False, **cfg.regress_artifact)
    out_files = dict()
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    out_files[in_key] = bids_path_in.copy().update(processing="regress")
    msg, _ = _read_raw_msg(bids_path_in=bids_path_in, run=run, task=task)
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
            title_prefix="Raw (regression)",
            tags=("regression",),
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
        regress_artifact=config.regress_artifact,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run artifact regression."""
    if config.regress_artifact is None:
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ssrt = _get_ssrt(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_regress_artifact, exec_params=config.exec_params, n_iter=len(ssrt)
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
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )

    save_logs(config=config, logs=logs)
