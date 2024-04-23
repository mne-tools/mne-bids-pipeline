"""Assess data quality and find bad (and flat) channels."""

from types import SimpleNamespace

import mne
import pandas as pd

from ..._config_utils import (
    _do_mf_autobad,
    _pl,
    get_mf_cal_fname,
    get_mf_ctc_fname,
    get_runs_tasks,
    get_sessions,
    get_subjects,
)
from ..._import_data import (
    _bads_path,
    _get_mf_reference_run_path,
    _get_run_rest_noise_path,
    _import_data_kwargs,
    _read_raw_msg,
    import_er_data,
    import_experimental_data,
)
from ..._io import _write_json
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _add_raw, _open_report
from ..._run import _prep_out_files, failsafe_run, save_logs
from ..._viz import plot_auto_scores


def get_input_fnames_data_quality(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> dict:
    """Get paths of files required by assess_data_quality function."""
    kwargs = dict(
        cfg=cfg,
        subject=subject,
        session=session,
        add_bads=False,
    )
    in_files = _get_run_rest_noise_path(
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        **kwargs,
    )
    # When doing autobad for the noise run, we also need the reference run
    if _do_mf_autobad(cfg=cfg) and run is None and task == "noise":
        in_files.update(_get_mf_reference_run_path(**kwargs))
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_data_quality,
)
def assess_data_quality(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: dict,
) -> dict:
    """Assess data quality and find and mark bad channels."""
    import matplotlib.pyplot as plt

    out_files = dict()
    key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(key)
    if key == "raw_task-noise_run-None":
        bids_path_ref_in = in_files.pop("raw_ref_run", None)
    else:
        bids_path_ref_in = None
    msg, _ = _read_raw_msg(bids_path_in=bids_path_in, run=run, task=task)
    logger.info(**gen_log_kwargs(message=msg))
    if run is None and task == "noise":
        raw = import_er_data(
            cfg=cfg,
            bids_path_er_in=bids_path_in,
            bids_path_er_bads_in=None,
            bids_path_ref_in=bids_path_ref_in,
            bids_path_ref_bads_in=None,
            prepare_maxwell_filter=True,
        )
    else:
        data_is_rest = run is None and task == "rest"
        raw = import_experimental_data(
            bids_path_in=bids_path_in,
            bids_path_bads_in=None,
            cfg=cfg,
            data_is_rest=data_is_rest,
        )
    preexisting_bads = sorted(raw.info["bads"])

    if _do_mf_autobad(cfg=cfg):
        (
            auto_noisy_chs,
            auto_flat_chs,
            auto_scores,
        ) = _find_bads_maxwell(
            cfg=cfg,
            exec_params=exec_params,
            raw=raw,
            subject=subject,
            session=session,
            run=run,
            task=task,
        )
        bads = sorted(set(raw.info["bads"] + auto_noisy_chs + auto_flat_chs))
        msg = f"Found {len(bads)} bad channel{_pl(bads)}."
        raw.info["bads"] = bads
        del bads
        logger.info(**gen_log_kwargs(message=msg))
    else:
        auto_scores = auto_noisy_chs = auto_flat_chs = None
    del key

    # Always output the scores and bads TSV
    out_files["auto_scores"] = bids_path_in.copy().update(
        suffix="scores",
        extension=".json",
        root=cfg.deriv_root,
        split=None,
        check=False,
        session=session,
        subject=subject,
    )
    _write_json(out_files["auto_scores"], auto_scores)

    # Write the bad channels to disk.
    out_files["bads_tsv"] = _bads_path(
        cfg=cfg,
        bids_path_in=bids_path_in,
        subject=subject,
        session=session,
    )
    bads_for_tsv = []
    reasons = []

    if auto_flat_chs:
        for ch in auto_flat_chs:
            reason = (
                "pre-existing (before MNE-BIDS-pipeline was run) & auto-flat"
                if ch in preexisting_bads
                else "auto-flat"
            )
            bads_for_tsv.append(ch)
            reasons.append(reason)

    if auto_noisy_chs:
        for ch in auto_noisy_chs:
            reason = (
                "pre-existing (before MNE-BIDS-pipeline was run) & auto-noisy"
                if ch in preexisting_bads
                else "auto-noisy"
            )
            bads_for_tsv.append(ch)
            reasons.append(reason)

    if preexisting_bads:
        for ch in preexisting_bads:
            if ch in bads_for_tsv:
                continue
            bads_for_tsv.append(ch)
            reasons.append("pre-existing (before MNE-BIDS-pipeline was run)")

    tsv_data = pd.DataFrame(dict(name=bads_for_tsv, reason=reasons))
    tsv_data = tsv_data.sort_values(by="name")
    tsv_data.to_csv(out_files["bads_tsv"], sep="\t", index=False)

    # Report
    # Restore bads to their original state so they will show up in the report
    raw.info["bads"] = preexisting_bads

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        # Original data
        kind = "original" if not cfg.proc else cfg.proc
        msg = f"Adding {kind} raw data to report"
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=bids_path_in,
            raw=raw,
            title=f"Raw ({kind})",
            tags=("data-quality",),
        )
        title = f"Bad channel detection: {run}"
        if cfg.find_noisy_channels_meg:
            assert auto_scores is not None
            msg = "Adding noisy channel detection to report"
            logger.info(**gen_log_kwargs(message=msg))
            figs = plot_auto_scores(auto_scores, ch_types=cfg.ch_types)
            captions = [f"Run {run}"] * len(figs)
            tags = ("raw", "data-quality", f"run-{run}")
            report.add_figure(
                fig=figs,
                caption=captions,
                section="Data quality",
                title=title,
                tags=tags,
                replace=True,
            )
            for fig in figs:
                plt.close(fig)
        else:
            report.remove(title=title)

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def _find_bads_maxwell(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
):
    if cfg.find_flat_channels_meg:
        if cfg.find_noisy_channels_meg:
            msg = "Finding flat channels and noisy channels using Maxwell filtering."
        else:
            msg = "Finding flat channels."
    else:
        assert cfg.find_noisy_channels_meg
        msg = "Finding noisy channels using Maxwell filtering."
    logger.info(**gen_log_kwargs(message=msg))

    # Filter the data manually before passing it to find_bad_channels_maxwell()
    # This reduces memory usage, as we can control the number of jobs used
    # during filtering.
    raw_filt = raw.copy().filter(l_freq=None, h_freq=40, n_jobs=1)
    (
        auto_noisy_chs,
        auto_flat_chs,
        auto_scores,
    ) = mne.preprocessing.find_bad_channels_maxwell(
        raw=raw_filt,
        calibration=cfg.mf_cal_fname,
        cross_talk=cfg.mf_ctc_fname,
        origin=cfg.mf_head_origin,
        coord_frame="head",
        return_scores=True,
        h_freq=None,  # we filtered manually above
    )
    del raw_filt

    if cfg.find_flat_channels_meg:
        if auto_flat_chs:
            msg = (
                f"Found {len(auto_flat_chs)} flat channels: "
                f'{", ".join(auto_flat_chs)}'
            )
        else:
            msg = "Found no flat channels."
        logger.info(**gen_log_kwargs(message=msg))
    else:
        auto_flat_chs = []

    if cfg.find_noisy_channels_meg:
        if auto_noisy_chs:
            msg = (
                f"Found {len(auto_noisy_chs)} noisy "
                f"channel{_pl(auto_noisy_chs)}: "
                f'{", ".join(auto_noisy_chs)}'
            )
        else:
            msg = "Found no noisy channels."

        logger.info(**gen_log_kwargs(message=msg))
    else:
        auto_noisy_chs = []

    # Interaction
    if exec_params.interactive and cfg.find_noisy_channels_meg:
        import matplotlib.pyplot as plt

        plot_auto_scores(auto_scores, ch_types=cfg.ch_types)
        plt.show()

    return auto_noisy_chs, auto_flat_chs, auto_scores


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    extra_kwargs = dict()
    if config.find_noisy_channels_meg or config.find_flat_channels_meg:
        # If these change, need to update hooks.py in doc build
        extra_kwargs["mf_cal_fname"] = get_mf_cal_fname(
            config=config,
            subject=subject,
            session=session,
        )
        extra_kwargs["mf_ctc_fname"] = get_mf_ctc_fname(
            config=config,
            subject=subject,
            session=session,
        )
        extra_kwargs["mf_head_origin"] = config.mf_head_origin
    cfg = SimpleNamespace(
        # These are included in _import_data_kwargs for automatic add_bads
        # detection
        # find_flat_channels_meg=config.find_flat_channels_meg,
        # find_noisy_channels_meg=config.find_noisy_channels_meg,
        **_import_data_kwargs(config=config, subject=subject),
        **extra_kwargs,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run assess_data_quality."""
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            assess_data_quality, exec_params=config.exec_params
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject, session=session),
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
