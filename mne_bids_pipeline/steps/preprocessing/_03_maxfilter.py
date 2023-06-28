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
    get_runs,
)
from ..._import_data import (
    import_experimental_data,
    import_er_data,
    _get_raw_paths,
    _add_rest_noise,
    _add_bads_file,
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
    in_files = _get_raw_paths(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="orig",
        add_bads=True,
    )
    if run == cfg.runs[0]:
        _add_rest_noise(
            cfg=cfg,
            subject=subject,
            session=session,
            in_files=in_files,
            kind="orig",
            add_bads=True,
        )
    # head positions
    if cfg.mf_mc:
        in_files[f"raw_run-{run}-pos"] = (
            in_files[f"raw_run-{run}"]
            .copy()
            .update(
                extension=".pos",
                root=cfg.deriv_root,
                check=False,
            )
        )
        rest_key = "raw_rest"
        if rest_key in in_files:
            in_files["raw_rest-pos"] = (
                in_files[rest_key]
                .copy()
                .update(
                    extension=".pos",
                    root=cfg.deriv_root,
                    check=False,
                )
            )
    # reference run (used for `destination` and also bad channels for noise)
    ref_bids_path = (
        list(in_files.values())[0]
        .copy()
        .update(
            run=cfg.mf_reference_run,
            check=True,
        )
    )
    key = "raw_ref_run"
    in_files[key] = ref_bids_path
    _add_bads_file(
        cfg=cfg,
        in_files=in_files,
        key=key,
    )
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
    in_key = f"raw_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads", None)
    bids_path_out = bids_path_in.copy().update(
        processing="sss",
        suffix="raw",
        extension=".fif",
        root=cfg.deriv_root,
        check=False,
    )
    # Now take everything from the bids_path_in and overwrite the parameters
    subject = bids_path_in.subject  # noqa: F841
    session = bids_path_in.session  # noqa: F841
    run = bids_path_in.run

    out_files = dict()
    # Load dev_head_t and digitization points from MaxFilter reference run.
    if cfg.mf_reference_run is not None:
        # Only log if we have more than just a single run
        msg = f"Loading reference run: {cfg.mf_reference_run or cfg.runs[0]}."
        logger.info(**gen_log_kwargs(message=msg))

    bids_path_ref_in = in_files.pop("raw_ref_run")
    raw = read_raw_bids(
        bids_path=bids_path_ref_in, extra_params=cfg.reader_extra_params
    )
    bids_path_ref_bads_in = in_files.pop("raw_ref_run-bads", None)
    if isinstance(destination, str):
        assert destination == "reference_run"
        destination = raw.info["dev_head_t"]
    del raw
    assert isinstance(destination, mne.transforms.Transform), destination

    raw = import_experimental_data(
        cfg=cfg,
        bids_path_in=bids_path_in,
        bids_path_bads_in=bids_path_bads_in,
        data_is_rest=False,
    )

    # Maxwell-filter experimental data.
    apply_msg = "Applying "

    if cfg.mf_st_duration:
        apply_msg += f"tSSS ({cfg.mf_st_duration} sec, corr={cfg.mf_st_correlation})"
    else:
        apply_msg += "SSS"
    if cfg.mf_mc:
        head_pos = mne.chpi.read_head_pos(in_files.pop(f"raw_run-{run}-pos"))
        apply_msg += " with MC"
    else:
        head_pos = None
    apply_msg += " to"
    logger.info(**gen_log_kwargs(message=f"{apply_msg} experimental data"))

    # Give some information about the transformation
    fr = raw.info["dev_head_t"]["trans"]
    to = destination["trans"]
    dist = 1000 * np.linalg.norm(fr[:3, 3] - to[:3, 3])
    angle = np.rad2deg(
        mne.transforms._angle_between_quats(
            *mne.transforms.rot_to_quat(np.array([to[:3, :3], fr[:3, :3]]))
        )
    )
    msg = (
        f"Destination head position is {dist:0.1f} mm and {angle:0.1f}° "
        "from dev_head_t"
    )
    logger.info(**gen_log_kwargs(message=msg))

    # Warn if no bad channels are set before Maxwell filter
    if not raw.info["bads"]:
        msg = (
            "No channels were marked as bad. Please carefully check "
            "your data to ensure this is correct; otherwise, Maxwell "
            "filtering WILL cause problems."
        )
        logger.warning(**gen_log_kwargs(message=msg))

    # Keyword arguments shared between Maxwell filtering of the
    # experimental and the empty-room data.
    common_mf_kws = dict(
        calibration=in_files.pop("mf_cal_fname"),
        cross_talk=in_files.pop("mf_ctc_fname"),
        st_duration=cfg.mf_st_duration,
        st_correlation=cfg.mf_st_correlation,
        origin=cfg.mf_head_origin,
        coord_frame="head",
        destination=destination,
    )

    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        head_pos=head_pos,
        **common_mf_kws,
    )
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
    del raw
    # we need to be careful about split files
    _update_for_splits(out_files, "sss_raw")

    if exec_params.interactive:
        raw_sss.plot(n_channels=50, butterfly=True, block=True)
    del raw_sss

    # Reporting
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, run=run
    ) as report:
        msg = "Adding Maxwell filtered raw data to report."
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files["sss_raw"],
            title="Raw (maxwell filtered)",
            tags=("sss",),
        )
    del bids_path_in

    # Noise data processing.
    nice_names = dict(rest="resting-state", noise="empty-room")
    for task in ("rest", "noise"):
        in_key = f"raw_{task}"
        if in_key not in in_files:
            continue
        recording_type = nice_names[task]
        msg = f"Processing {recording_type} recording …"
        logger.info(**gen_log_kwargs(message=msg, run=task))
        bids_path_noise = in_files.pop(in_key)
        bids_path_noise_bads = in_files.pop(f"{in_key}-bads", None)
        use_head_pos = head_pos
        if task == "rest":
            raw_noise = import_experimental_data(
                cfg=cfg,
                bids_path_in=bids_path_noise,
                bids_path_bads_in=bids_path_noise_bads,
                data_is_rest=True,
            )
            if cfg.mf_mc:
                use_head_pos = mne.chpi.read_head_positions(
                    in_files.pop("raw_rest-pos")
                )
        else:
            raw_noise = import_er_data(
                cfg=cfg,
                bids_path_er_in=bids_path_noise,
                bids_path_ref_in=bids_path_ref_in,
                # TODO: This can break processing, need to use union for all,
                # otherwise can get for ds003392:
                # "Reference run data rank does not match empty-room data rank"
                # bids_path_er_bads_in=bids_path_noise_bads,
                bids_path_er_bads_in=None,
                bids_path_ref_bads_in=bids_path_ref_bads_in,
                prepare_maxwell_filter=True,
            )

        # Maxwell-filter noise data.
        msg = f"{apply_msg} {recording_type} data"
        logger.info(**gen_log_kwargs(message=msg, run=task))
        raw_noise_sss = mne.preprocessing.maxwell_filter(
            raw_noise,
            head_pos=use_head_pos,
            **common_mf_kws,
        )
        if filter_chpi:
            logger.info(**gen_log_kwargs(message="Filtering cHPI", run=task))
            # allow_line_only=True is really mostly for the "noise" run
            mne.chpi.filter_chpi(raw_noise_sss, allow_line_only=True)

        # Perform a sanity check: empty-room rank should exactly match the
        # experimental data rank after Maxwell filtering; resting-state rank
        # should be equal or be greater than experimental data rank.
        #
        # We're treating the two cases differently, because we don't
        # copy the bad channel selection from the reference run over to
        # the resting-state recording.

        raw_sss = mne.io.read_raw_fif(out_files["sss_raw"])
        rank_exp = mne.compute_rank(raw_sss, rank="info")["meg"]
        rank_noise = mne.compute_rank(raw_noise_sss, rank="info")["meg"]

        if task == "rest":
            if rank_exp > rank_noise:
                msg = (
                    f"Resting-state rank ({rank_noise}) is lower than "
                    f"reference run data rank ({rank_exp}). We will try to "
                    f"take care of this during epoching of the experimental "
                    f"data."
                )
                logger.warning(**gen_log_kwargs(message=msg, run=task))
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

        out_files[in_key] = bids_path_out.copy().update(
            task=task, run=None, processing="sss"
        )

        # Save only the channel types we wish to analyze
        # (same as for experimental data above).
        msg = "Writing " f"{out_files[in_key].fpath.relative_to(cfg.deriv_root)}"
        logger.info(**gen_log_kwargs(message=msg, run=task))
        raw_noise_sss.save(
            out_files[in_key],
            overwrite=True,
            split_naming="bids",
            split_size=cfg._raw_split_size,
        )
        _update_for_splits(out_files, in_key)
        del raw_noise_sss

        with _open_report(
            cfg=cfg, exec_params=exec_params, subject=subject, session=session, run=task
        ) as report:
            msg = "Adding Maxwell filtered raw data to report"
            logger.info(**gen_log_kwargs(message=msg, run=task))
            _add_raw(
                cfg=cfg,
                report=report,
                bids_path_in=out_files[in_key],
                title="Raw (maxwell filtered)",
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
                task=None,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for run in get_runs(config=config, subject=subject)
        )

    save_logs(config=config, logs=logs)
