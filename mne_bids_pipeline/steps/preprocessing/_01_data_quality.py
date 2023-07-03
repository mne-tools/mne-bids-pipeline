"""Assess data quality and find bad (and flat) channels."""

from types import SimpleNamespace
from typing import Optional

import pandas as pd

import mne
from mne.utils import _pl
from mne_bids import BIDSPath

from ..._config_utils import (
    get_mf_cal_fname,
    get_mf_ctc_fname,
    get_subjects,
    get_sessions,
    get_runs_tasks,
    _do_mf_autobad,
)
from ..._import_data import (
    _get_run_rest_noise_path,
    _get_mf_reference_run_path,
    import_experimental_data,
    import_er_data,
    _bads_path,
    _auto_scores_path,
    _import_data_kwargs,
)
from ..._io import _write_json
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report, _add_raw
from ..._run import failsafe_run, save_logs
from ..._viz import plot_auto_scores


def get_input_fnames_data_quality(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
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
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
    in_files: dict,
) -> dict:
    """Assess data quality and find and mark bad channels."""
    import matplotlib.pyplot as plt

    out_files = dict()
    key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(key)
    if _do_mf_autobad(cfg=cfg):
        if key == "raw_task-noise_run-None":
            bids_path_ref_in = in_files.pop("raw_ref_run")
        else:
            bids_path_ref_in = None
        auto_scores = _find_bads_maxwell(
            cfg=cfg,
            exec_params=exec_params,
            bids_path_in=bids_path_in,
            bids_path_ref_in=bids_path_ref_in,
            subject=subject,
            session=session,
            run=run,
            task=task,
            out_files=out_files,
        )
    else:
        auto_scores = None
    del key

    # Report
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
            title=f"Raw ({kind})",
            tags=("data-quality",),
        )
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
                title=f"Bad channel detection: {run}",
                tags=tags,
                replace=True,
            )
            for fig in figs:
                plt.close(fig)

    assert len(in_files) == 0, in_files.keys()
    return out_files


def _find_bads_maxwell(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    bids_path_in: BIDSPath,
    bids_path_ref_in: Optional[BIDSPath],
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
    out_files: dict,
):
    if cfg.find_flat_channels_meg and not cfg.find_noisy_channels_meg:
        msg = "Finding flat channels."
    elif cfg.find_noisy_channels_meg and not cfg.find_flat_channels_meg:
        msg = "Finding noisy channels using Maxwell filtering."
    else:
        msg = "Finding flat channels and noisy channels using " "Maxwell filtering."
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

    # Filter the data manually before passing it to find_bad_channels_maxwell()
    # This reduces memory usage, as we can control the number of jobs used
    # during filtering.
    preexisting_bads = raw.info["bads"].copy()
    bads = preexisting_bads.copy()
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
        bads.extend(auto_flat_chs)

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
        bads.extend(auto_noisy_chs)

    bads = sorted(set(bads))
    msg = f"Found {len(bads)} channel{_pl(bads)} as bad."
    raw.info["bads"] = bads
    del bads
    logger.info(**gen_log_kwargs(message=msg))

    if cfg.find_noisy_channels_meg:
        out_files["auto_scores"] = _auto_scores_path(
            cfg=cfg,
            bids_path_in=bids_path_in,
        )
        if not out_files["auto_scores"].fpath.parent.exists():
            out_files["auto_scores"].fpath.parent.mkdir(parents=True)
        _write_json(out_files["auto_scores"], auto_scores)

    # Write the bad channels to disk.
    out_files["bads_tsv"] = _bads_path(
        cfg=cfg,
        bids_path_in=bids_path_in,
    )
    bads_for_tsv = []
    reasons = []

    if cfg.find_flat_channels_meg:
        bads_for_tsv.extend(auto_flat_chs)
        reasons.extend(["auto-flat"] * len(auto_flat_chs))
        preexisting_bads = set(preexisting_bads) - set(auto_flat_chs)

    if cfg.find_noisy_channels_meg:
        bads_for_tsv.extend(auto_noisy_chs)
        reasons.extend(["auto-noisy"] * len(auto_noisy_chs))
        preexisting_bads = set(preexisting_bads) - set(auto_noisy_chs)

    preexisting_bads = list(preexisting_bads)
    if preexisting_bads:
        bads_for_tsv.extend(preexisting_bads)
        reasons.extend(
            ["pre-existing (before MNE-BIDS-pipeline was run)"] * len(preexisting_bads)
        )

    tsv_data = pd.DataFrame(dict(name=bads_for_tsv, reason=reasons))
    tsv_data = tsv_data.sort_values(by="name")
    tsv_data.to_csv(out_files["bads_tsv"], sep="\t", index=False)

    # Interaction
    if exec_params.interactive and cfg.find_noisy_channels_meg:
        import matplotlib.pyplot as plt

        plot_auto_scores(auto_scores, ch_types=cfg.ch_types)
        plt.show()

    return auto_scores


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> SimpleNamespace:
    extra_kwargs = dict()
    if config.find_noisy_channels_meg or config.find_flat_channels_meg:
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
    """Run maxwell_filter."""
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
