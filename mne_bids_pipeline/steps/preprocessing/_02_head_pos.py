"""Estimate head positions."""

from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import _get_ss, _get_ssrt, get_runs_tasks
from mne_bids_pipeline._import_data import (
    _get_bids_path_in,
    _get_run_rest_noise_path,
    _import_data_kwargs,
    _path_dict,
    import_experimental_data,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _get_prefix_tags, _open_report
from mne_bids_pipeline._run import _prep_out_files, failsafe_run, save_logs
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_head_pos(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by run_head_pos function."""
    return _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
        add_bads=True,
    )


@failsafe_run(
    get_input_fnames=get_input_fnames_head_pos,
)
def run_head_pos(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads")
    out_files = dict()
    key = f"raw_run-{run}-pos"
    out_files[key] = bids_path_in.copy().update(
        suffix="headpos",
        extension=".txt",
        root=cfg.deriv_root,
        check=False,
    )
    # Now take everything from the bids_path_in and overwrite the parameters
    assert bids_path_in.subject is not None
    subject: str = bids_path_in.subject  # noqa: F841
    session: str | None = bids_path_in.session  # noqa: F841
    run = bids_path_in.run

    raw = import_experimental_data(
        cfg=cfg,
        bids_path_in=bids_path_in,
        bids_path_bads_in=bids_path_bads_in,
        data_is_rest=None,  # autodetect
    )
    # TODO: We should split these into separate cached steps.
    # This could all be part of this file -- we can put multiple loops inside
    # main() to do it. But this can wait until a use case pushes us to do it.
    logger.info(**gen_log_kwargs(message="Estimating cHPI amplitudes"))
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(
        raw,
        t_step_min=cfg.mf_mc_t_step_min,
        t_window=cfg.mf_mc_t_window,
    )
    logger.info(**gen_log_kwargs(message="Estimating cHPI SNR"))
    snr_dict = mne.chpi.compute_chpi_snr(
        raw,
        t_step_min=cfg.mf_mc_t_step_min,
        t_window=cfg.mf_mc_t_window,
    )
    logger.info(**gen_log_kwargs(message="Estimating cHPI locations"))
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    logger.info(**gen_log_kwargs(message="Estimating head positions"))
    head_pos = mne.chpi.compute_head_pos(
        raw.info,
        chpi_locs,
        gof_limit=cfg.mf_mc_gof_limit,
        dist_limit=cfg.mf_mc_dist_limit,
    )
    mne.chpi.write_head_pos(out_files[key], head_pos)

    # Reporting
    prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=task, run=run)
    tags = ("raw", "chpi", "sss") + extra_tags
    section = "Head position"

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Adding cHPI SNR and head positions to report."
        logger.info(**gen_log_kwargs(message=msg))
        fig = mne.viz.plot_chpi_snr(snr_dict)
        report.add_figure(
            fig=fig,
            title=f"cHPI SNR{prefix}",
            image_format="svg",
            section=section,
            tags=tags,
            replace=True,
        )
        plt.close(fig)
        fig = mne.viz.plot_head_positions(head_pos, mode="traces")
        report.add_figure(
            fig=fig,
            title=f"Head positions{prefix}",
            image_format="svg",
            section=section,
            tags=tags,
            replace=True,
        )
        plt.close(fig)
    del bids_path_in
    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_input_fnames_twa_head_pos(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> dict[str, BIDSPath]:
    """Get paths of files required by compute_twa_head_pos function."""
    in_files: dict[str, BIDSPath] = dict()
    # can't use `_get_run_path()` here because we don't loop over runs/tasks.
    # But any run will do, as long as the file exists:
    runs_tasks = cfg.runs_tasks
    for run, task in runs_tasks:
        bids_path_in = _get_bids_path_in(
            cfg=cfg,
            subject=subject,
            session=session,
            run=run,
            task=task,
            kind="orig",
        )
        in_files.update(
            _path_dict(
                cfg=cfg,
                subject=subject,
                session=session,
                bids_path_in=bids_path_in,
                add_bads=False,
                allow_missing=False,
                kind="orig",
            )
        )
        # and headpos
        key_end = f"task-{task}_run-{run}"
        in_files[f"headpos_{key_end}"] = (
            in_files[f"raw_{key_end}"]
            .copy()
            .update(
                suffix="headpos",
                extension=".txt",
                root=cfg.deriv_root,
            )
        )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_twa_head_pos,
)
def compute_twa_head_pos(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | list[str] | None,
    in_files: InFilesT,
) -> OutFilesT:
    """Compute time-weighted average head position."""
    # logging
    want_mc = cfg.mf_mc
    dest_is_twa = isinstance(cfg.mf_destination, str) and cfg.mf_destination == "twa"
    msg = "Skipping, "
    if not want_mc:
        msg += " no movement compensation requested …"
    elif not dest_is_twa:
        msg += ' mf_destination is not "twa" …'
    else:
        msg = "Computing time-weighted average head position"
    logger.info(**gen_log_kwargs(message=msg))
    # maybe bail early
    if not want_mc and not dest_is_twa:
        return _prep_out_files(exec_params=exec_params, out_files=dict())

    # path to (subject+session)-level `destination.fif` in derivatives folder
    dest_path = (
        list(in_files.values())[0]
        .copy()
        .update(
            check=False,
            description="twa",
            extension=".fif",
            root=cfg.deriv_root,
            run=None,
            task=None,
            suffix="destination",
        )
    )
    # need raw files from all runs
    raw_fnames = [in_files[key].fpath for key in in_files if key.startswith("raw")]
    raws = [
        mne.io.read_raw_fif(fname, allow_maxshield=True, verbose="ERROR", preload=False)
        for fname in raw_fnames
    ]
    head_poses = [
        mne.chpi.read_head_pos(in_files[key].fpath)
        for key in in_files
        if key.startswith("headpos")
    ]
    in_files.clear()
    # compute time-weighted average head position and save it to disk
    destination = mne.preprocessing.compute_average_dev_head_t(raws, head_poses)
    mne.write_trans(fname=dest_path.fpath, trans=destination, overwrite=True)
    # output
    out_files = dict(destination_head_pos=dest_path)
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config_head_pos(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        mf_mc_t_step_min=config.mf_mc_t_step_min,
        mf_mc_gof_limit=config.mf_mc_gof_limit,
        mf_mc_dist_limit=config.mf_mc_dist_limit,
        mf_mc_t_window=config.mf_mc_t_window,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def get_config_twa(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        runs_tasks=get_runs_tasks(
            config=config,
            subject=subject,
            session=session,
            which=("runs",),
        ),
        mf_mc=config.mf_mc,
        mf_destination=config.mf_destination,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run head position estimation."""
    if not config.use_maxwell_filter or not config.mf_mc:
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ss = _get_ss(config=config)
    ssrt = _get_ssrt(config=config, which=("runs", "rest"))
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_head_pos,
            exec_params=config.exec_params,
            n_iter=len(ssrt),
        )
        logs = parallel(
            run_func(
                cfg=get_config_head_pos(
                    config=config, subject=subject, session=session
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )
        # compute time-weighted average head position
        # within subject+session+task, across runs
        parallel, run_func = parallel_func(
            compute_twa_head_pos,
            exec_params=config.exec_params,
            n_iter=len(ss),
        )
        more_logs = parallel(
            run_func(
                cfg=get_config_twa(config=config, subject=subject, session=session),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject, session in ss
        )

    save_logs(config=config, logs=logs + more_logs)
