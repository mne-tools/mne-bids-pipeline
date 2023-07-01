"""Maxwell-filter MEG data.

If you chose to run Maxwell filter (config.use_maxwell_filter = True),
the data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files.
"""

import gc
from typing import Optional
from types import SimpleNamespace

import numpy as np
import mne
from mne_bids import read_raw_bids

from ..._config_utils import (
    get_mf_cal_fname,
    get_mf_ctc_fname,
    get_subjects,
    get_sessions,
    get_runs_tasks,
)
from ..._import_data import (
    import_experimental_data,
    import_er_data,
    _get_run_path,
    _get_run_rest_noise_path,
    _get_mf_reference_run_path,
    _import_data_kwargs,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report, _add_raw
from ..._run import failsafe_run, save_logs, _update_for_splits


def get_input_fnames_maxwell_filter(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
) -> dict:
    """Get paths of files required by maxwell_filter function."""
    kwargs = dict(
        cfg=cfg,
        subject=subject,
        session=session,
    )
    in_files = _get_run_rest_noise_path(
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        **kwargs,
    )
    # head positions
    if cfg.mf_mc:
        if run is None and task == "noise":
            pos_run, pos_task = cfg.mf_reference_run, cfg.task
        else:
            pos_run, pos_task = run, task
        path = _get_run_path(
            run=pos_run,
            task=pos_task,
            add_bads=False,
            kind="orig",
            **kwargs,
        )[f"raw_task-{pos_task}_run-{pos_run}"]
        in_files[f"raw_task-{task}_run-{run}-pos"] = path.update(
            extension=".pos",
            root=cfg.deriv_root,
            check=False,
            task=pos_task,
            run=pos_run,
        )

    # reference run (used for `destination` and also bad channels for noise)
    in_files.update(_get_mf_reference_run_path(add_bads=True, **kwargs))

    # standard files
    in_files["mf_cal_fname"] = cfg.mf_cal_fname
    in_files["mf_ctc_fname"] = cfg.mf_ctc_fname
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_maxwell_filter,
)
def run_maxwell_filter(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    task: Optional[str],
    in_files: dict,
) -> dict:
    if cfg.proc and "sss" in cfg.proc and cfg.use_maxwell_filter:
        raise ValueError(
            f"You cannot set use_maxwell_filter to True "
            f"if data have already processed with Maxwell-filter."
            f" Got proc={cfg.proc}."
        )
    if isinstance(cfg.mf_destination, str):
        destination = cfg.mf_destination
        assert destination == "reference_run"
    else:
        destination = np.array(cfg.mf_destination, float)
        assert destination.shape == (4, 4)
        destination = mne.transforms.Transform("meg", "head", destination)

    filter_chpi = cfg.mf_mc if cfg.mf_filter_chpi is None else cfg.mf_filter_chpi
    is_rest_noise = run is None and task in ("noise", "rest")
    if is_rest_noise:
        nice_names = dict(rest="resting-state", noise="empty-room")
        recording_type = nice_names[task]
    else:
        recording_type = "experimental"
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads", None)
    bids_path_out_kwargs = dict(
        subject=subject,  # need these in the case of an empty room match
        session=session,
        processing="sss",
        suffix="raw",
        extension=".fif",
        root=cfg.deriv_root,
        check=False,
    )
    bids_path_out = bids_path_in.copy().update(**bids_path_out_kwargs)

    # Now take everything from the bids_path_in and overwrite the parameters
    subject = bids_path_in.subject  # noqa: F841
    session = bids_path_in.session  # noqa: F841
    run = bids_path_in.run

    out_files = dict()
    # Load dev_head_t and digitization points from MaxFilter reference run.
    msg = f"Loading reference run: {cfg.mf_reference_run}."
    logger.info(**gen_log_kwargs(message=msg))

    bids_path_ref_in = in_files.pop("raw_ref_run")
    raw = read_raw_bids(
        bids_path=bids_path_ref_in,
        extra_params=cfg.reader_extra_params,
        verbose=cfg.read_raw_bids_verbose,
    )
    bids_path_ref_bads_in = in_files.pop("raw_ref_run-bads", None)
    if isinstance(destination, str):
        assert destination == "reference_run"
        destination = raw.info["dev_head_t"]
    del raw
    assert isinstance(destination, mne.transforms.Transform), destination

    # Maxwell-filter experimental data.
    apply_msg = "Applying "
    if cfg.mf_st_duration:
        apply_msg += f"tSSS ({cfg.mf_st_duration} sec, corr={cfg.mf_st_correlation})"
    else:
        apply_msg += "SSS"
    if cfg.mf_mc:
        apply_msg += " with MC"
        head_pos = mne.chpi.read_head_pos(in_files.pop(f"{in_key}-pos"))
    else:
        head_pos = None
    apply_msg += " to"

    mf_kws = dict(
        calibration=in_files.pop("mf_cal_fname"),
        cross_talk=in_files.pop("mf_ctc_fname"),
        st_duration=cfg.mf_st_duration,
        st_correlation=cfg.mf_st_correlation,
        origin=cfg.mf_head_origin,
        coord_frame="head",
        destination=destination,
        head_pos=head_pos,
    )

    logger.info(**gen_log_kwargs(message=f"{apply_msg} {recording_type} data"))
    if not (run is None and task == "noise"):
        data_is_rest = run is None and task == "rest"
        raw = import_experimental_data(
            cfg=cfg,
            bids_path_in=bids_path_in,
            bids_path_bads_in=bids_path_bads_in,
            data_is_rest=data_is_rest,
        )
        fr = raw.info["dev_head_t"]["trans"]
        where = "original head position"
    else:
        raw = import_er_data(
            cfg=cfg,
            bids_path_er_in=bids_path_in,
            bids_path_ref_in=bids_path_ref_in,
            # TODO: This can break processing, need to use union for all,
            # otherwise can get for ds003392:
            # "Reference run data rank does not match empty-room data rank"
            # bids_path_er_bads_in=bids_path_noise_bads,
            bids_path_er_bads_in=None,
            bids_path_ref_bads_in=bids_path_ref_bads_in,
            prepare_maxwell_filter=True,
        )
        fr = np.eye(4)
        where = "MEG device origin"

    # Give some information about the transformation
    to = destination["trans"]
    dist = 1000 * np.linalg.norm(fr[:3, 3] - to[:3, 3])
    angle = np.rad2deg(
        mne.transforms._angle_between_quats(
            *mne.transforms.rot_to_quat(np.array([to[:3, :3], fr[:3, :3]]))
        )
    )
    msg = f"Destination is {dist:0.1f} mm and {angle:0.1f}° from the {where}"
    logger.info(**gen_log_kwargs(message=msg))

    # Warn if no bad channels are set before Maxwell filter
    if not raw.info["bads"]:
        msg = (
            "No channels were marked as bad. Please carefully check "
            "your data to ensure this is correct; otherwise, Maxwell "
            "filtering WILL cause problems."
        )
        logger.warning(**gen_log_kwargs(message=msg))

    raw_sss = mne.preprocessing.maxwell_filter(raw, **mf_kws)
    del raw
    gc.collect()

    if is_rest_noise:
        # Perform a sanity check: empty-room rank should exactly match the
        # experimental data rank after Maxwell filtering; resting-state rank
        # should be equal or be greater than experimental data rank.
        #
        # We're treating the two cases differently, because we don't
        # copy the bad channel selection from the reference run over to
        # the resting-state recording.

        bids_path_ref_sss = bids_path_ref_in.copy().update(**bids_path_out_kwargs)
        raw_exp = mne.io.read_raw_fif(bids_path_ref_sss)
        rank_exp = mne.compute_rank(raw_exp, rank="info")["meg"]
        rank_noise = mne.compute_rank(raw_sss, rank="info")["meg"]
        del raw_exp

        if task == "rest":
            if rank_exp > rank_noise:
                msg = (
                    f"Resting-state rank ({rank_noise}) is lower than "
                    f"reference run data rank ({rank_exp}). We will try to "
                    f"take care of this during epoching of the experimental "
                    f"data."
                )
                logger.warning(**gen_log_kwargs(message=msg))
            else:
                pass  # Should cause no problems!
        elif not np.isclose(rank_exp, rank_noise):
            msg = (
                f"Reference run data rank {rank_exp:.1f} does not "
                f"match {recording_type} data rank {rank_noise:.1f} after "
                f"Maxwell filtering. This indicates that the data "
                f"were processed  differently."
            )
            raise RuntimeError(msg)

    if filter_chpi:
        logger.info(**gen_log_kwargs(message="Filtering cHPI"))
        mne.chpi.filter_chpi(
            raw_sss,
            t_window=cfg.mf_mc_t_window,
        )

    out_files["sss_raw"] = bids_path_out
    msg = f"Writing {out_files['sss_raw'].fpath.relative_to(cfg.deriv_root)}"
    logger.info(**gen_log_kwargs(message=msg))
    raw_sss.save(
        out_files["sss_raw"],
        split_naming="bids",
        overwrite=True,
        split_size=cfg._raw_split_size,
    )
    _update_for_splits(out_files, "sss_raw")

    if exec_params.interactive:
        raw_sss.plot(n_channels=50, butterfly=True, block=True)

    # Reporting
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Adding Maxwell filtered raw data to report."
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files["sss_raw"],
            title="Raw (maxwell filtered)",
            tags=("sss",),
            raw=raw_sss,
        )

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        mf_cal_fname=get_mf_cal_fname(
            config=config,
            subject=subject,
            session=session,
        ),
        mf_ctc_fname=get_mf_ctc_fname(
            config=config,
            subject=subject,
            session=session,
        ),
        mf_st_duration=config.mf_st_duration,
        mf_st_correlation=config.mf_st_correlation,
        mf_head_origin=config.mf_head_origin,
        mf_mc=config.mf_mc,
        mf_filter_chpi=config.mf_filter_chpi,
        mf_destination=config.mf_destination,
        mf_int_order=config.mf_int_order,
        mf_mc_t_window=config.mf_mc_t_window,
        **_import_data_kwargs(config=config, subject=subject),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run maxwell_filter."""
    if not config.use_maxwell_filter:
        msg = "Skipping …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_maxwell_filter, exec_params=config.exec_params
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
