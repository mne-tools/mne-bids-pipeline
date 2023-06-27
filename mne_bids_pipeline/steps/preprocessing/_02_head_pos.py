"""Estimate head positions."""

from typing import Optional
from types import SimpleNamespace

import mne

from ..._config_utils import (
    get_subjects,
    get_sessions,
    get_runs_tasks,
)
from ..._import_data import (
    import_experimental_data,
    _get_raw_paths,
    _import_data_kwargs,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report
from ..._run import failsafe_run, save_logs


def get_input_fnames_head_pos(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
) -> dict:
    """Get paths of files required by run_head_pos function."""
    # TODO: This is a an ugly hack -- we trick _get_raw_paths into
    # thinking the run=None task="rest" case is really runs[0] so it adds
    # that raw file *plus* the rest case... then remove the runs[0] key.
    # This should be refactored at some point...
    if run is None and task == "rest":
        use_run = cfg.runs[0]
    else:
        use_run = run
    in_files = _get_raw_paths(
        cfg=cfg,
        subject=subject,
        session=session,
        run=use_run,
        kind="orig",
        add_bads=True,
        include_mf_ref=False,
    )
    # ... finally remove the shim to get the right rest path
    remove_keys = list()
    if run is None and task == "rest":
        remove_keys.append(f"raw_run-{run}")
    # ... and remove the raw_noise that might have been added
    remove_keys.append("raw_noise")
    for key in remove_keys:
        if key in in_files:
            in_files.pop(key)
            in_files.pop(f"{key}-bads", None)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_head_pos,
)
def run_head_pos(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
    in_files: dict,
) -> dict:
    import matplotlib.pyplot as plt

    if run is None and task == "rest":
        in_key = "raw_rest"
    else:
        in_key = f"raw_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads", None)
    out_files = dict()
    key = f"raw_run-{run}-pos"
    out_files[key] = bids_path_in.copy().update(
        extension=".pos",
        root=cfg.deriv_root,
        check=False,
    )
    # Now take everything from the bids_path_in and overwrite the parameters
    subject = bids_path_in.subject  # noqa: F841
    session = bids_path_in.session  # noqa: F841
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
    )
    logger.info(**gen_log_kwargs(message="Estimating cHPI SNR"))
    snr_dict = mne.chpi.compute_chpi_snr(
        raw,
        t_step_min=cfg.mf_mc_t_step_min,
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
    tags = ("raw", f"run-{bids_path_in.run}", "chpi", "sss")
    section = "Head position"
    title = f"run {bids_path_in.run}"

    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, run=run
    ) as report:
        msg = "Adding cHPI SNR and head positions to report."
        logger.info(**gen_log_kwargs(message=msg))
        fig = mne.viz.plot_chpi_snr(snr_dict)
        report.add_figure(
            fig=fig,
            title=f"cHPI SNR: {title}",
            image_format="svg",
            section=section,
            tags=tags,
            replace=True,
        )
        plt.close(fig)
        fig = mne.viz.plot_head_positions(head_pos, mode="traces")
        report.add_figure(
            fig=fig,
            title=f"Head positions: {title}",
            image_format="svg",
            section=section,
            tags=tags,
            replace=True,
        )
        plt.close(fig)
    del bids_path_in
    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        mf_mc_t_step_min=config.mf_mc_t_step_min,
        mf_mc_gof_limit=config.mf_mc_gof_limit,
        mf_mc_dist_limit=config.mf_mc_dist_limit,
        **_import_data_kwargs(config=config, subject=subject),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run head position estimation."""
    if not config.use_maxwell_filter or not config.mf_mc:
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(run_head_pos, exec_params=config.exec_params)
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
