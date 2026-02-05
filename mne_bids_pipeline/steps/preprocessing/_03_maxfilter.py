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
from copy import deepcopy
from types import SimpleNamespace

import mne
import numpy as np
from mne_bids import read_raw_bids

from mne_bids_pipeline._config_utils import (
    _get_ss,
    _get_ssrt,
    _pl,
    get_mf_cal_fname,
    get_mf_ctc_fname,
)
from mne_bids_pipeline._import_data import (
    _get_mf_reference_path,
    _get_run_path,
    _get_run_rest_noise_path,
    _import_data_kwargs,
    import_er_data,
    import_experimental_data,
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


# %% eSSS
def get_input_fnames_esss(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> InFilesT:
    in_files = _get_run_rest_noise_path(
        run=None,
        task="noise",
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
        cfg=cfg,
        subject=subject,
        session=session,
    )
    in_files.update(_get_mf_reference_path(cfg=cfg, subject=subject, session=session))
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_esss,
)
def compute_esss_proj(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    run, task = None, "noise"
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads", None)  # noqa
    bids_path_ref_in = in_files.pop("raw_ref_run")
    bids_path_ref_bads_in = in_files.pop("raw_ref_run-bads", None)
    raw_noise = import_er_data(
        cfg=cfg,
        bids_path_er_in=bids_path_in,
        bids_path_ref_in=bids_path_ref_in,
        # TODO: This must match below, so we don't pass it
        # bids_path_er_bads_in=bids_path_bads_in,
        bids_path_er_bads_in=None,
        bids_path_ref_bads_in=bids_path_ref_bads_in,
        prepare_maxwell_filter=True,
    )
    logger.info(
        **gen_log_kwargs(
            f"Computing eSSS basis with {cfg.mf_esss} component{_pl(cfg.mf_esss)}"
        )
    )
    projs = mne.compute_proj_raw(
        raw_noise,
        n_grad=cfg.mf_esss,
        n_mag=cfg.mf_esss,
        reject=cfg.mf_esss_reject,
        meg="combined",
    )
    msg = "Got variance explained of: " + ", ".join(
        f"{p['explained_var'] * 100:.1f}%" for p in projs
    )
    logger.info(**gen_log_kwargs(message=msg))
    out_files = dict()
    out_files["esss_basis"] = bids_path_in.copy().update(
        subject=subject,  # need these in the case of an empty room match
        session=session,
        run=run,
        task=task,
        suffix="esssproj",
        split=None,
        extension=".fif",
        root=cfg.deriv_root,
        check=False,
    )
    mne.write_proj(out_files["esss_basis"], projs, overwrite=True)

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Adding eSSS projectors to report."
        logger.info(**gen_log_kwargs(message=msg))
        kinds_picks = list()
        for kind in ("mag", "grad"):
            picks = mne.pick_types(raw_noise.info, meg=kind, exclude="bads")
            if not len(picks):
                continue
            kinds_picks.append([kind, picks])
        n_row, n_col = len(kinds_picks), cfg.mf_esss
        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(n_col + 0.5, n_row + 0.5),
            constrained_layout=True,
            squeeze=False,
        )
        # TODO: plot_projs_topomap doesn't handle meg="combined" well:
        # https://github.com/mne-tools/mne-python/pull/11792
        for ax_row, (kind, picks) in zip(axes, kinds_picks):
            info = mne.pick_info(raw_noise.info, picks)
            ch_names = info["ch_names"]
            these_projs = deepcopy(projs)
            for proj in these_projs:
                sub_idx = [proj["data"]["col_names"].index(name) for name in ch_names]
                proj["data"]["data"] = proj["data"]["data"][:, sub_idx]
                proj["data"]["col_names"] = ch_names
            mne.viz.plot_projs_topomap(
                these_projs,
                info=info,
                axes=ax_row,
            )
            for ai, ax in enumerate(ax_row):
                ax.set_title(f"{kind} {ai + 1}")
        report.add_figure(
            fig,
            title="eSSS projectors",
            tags=("sss", "raw"),
            replace=True,
        )
        plt.close(fig)

    return _prep_out_files(exec_params=exec_params, out_files=out_files)


# %% maxwell_filter


def get_input_fnames_maxwell_filter(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by maxwell_filter function."""
    in_files = _get_run_rest_noise_path(
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
        cfg=cfg,
        subject=subject,
        session=session,
    )
    in_key = f"raw_task-{task}_run-{run}"
    assert in_key in in_files
    # head positions
    if cfg.mf_mc:
        if run is None and task == "noise":
            pos_run, pos_task = cfg.mf_reference_run, cfg.mf_reference_task
        else:
            pos_run, pos_task = run, task
        path = _get_run_path(
            run=pos_run,
            task=pos_task,
            add_bads=False,
            kind="orig",
            cfg=cfg,
            subject=subject,
            session=session,
        )[f"raw_task-{pos_task}_run-{pos_run}"]
        in_files[f"{in_key}-pos"] = path.copy().update(
            suffix="headpos",
            extension=".txt",
            root=cfg.deriv_root,
            check=False,
            task=pos_task,
            run=pos_run,
        )
        if isinstance(cfg.mf_destination, str) and cfg.mf_destination == "twa":
            in_files[f"{in_key}-twa"] = path.update(
                description="twa",
                suffix="destination",
                extension=".fif",
                root=cfg.deriv_root,
                check=False,
                task=None,
                run=None,
            )
    if cfg.mf_esss:
        in_files["esss_basis"] = (
            in_files[in_key]
            .copy()
            .update(
                subject=subject,
                session=session,
                run=None,
                task="noise",
                suffix="esssproj",
                split=None,
                extension=".fif",
                root=cfg.deriv_root,
                check=False,
            )
        )

    # reference run (used for `destination` and also bad channels for noise)
    # use add_bads=None here to mean "add if autobad is turned on"
    in_files.update(_get_mf_reference_path(cfg=cfg, subject=subject, session=session))

    is_rest_noise = run is None and task in ("noise", "rest")
    if is_rest_noise:
        key = "raw_ref_run_sss"
        in_files[key] = (
            in_files["raw_ref_run"]
            .copy()
            .update(
                processing="sss",
                suffix="raw",
                extension=".fif",
                root=cfg.deriv_root,
                check=False,
            )
        )
        _update_for_splits(in_files, key, single=True)

    # set calibration and crosstalk files (if provided)
    if cfg.mf_cal_fname is not None:
        in_files["mf_cal_fname"] = cfg.mf_cal_fname
    if cfg.mf_ctc_fname is not None:
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
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    if cfg.proc and "sss" in cfg.proc and cfg.use_maxwell_filter:
        raise ValueError(
            f"You cannot set use_maxwell_filter to True "
            f"if data have already processed with Maxwell-filter."
            f" Got proc={cfg.proc}."
        )
    if isinstance(cfg.mf_destination, str):
        destination = cfg.mf_destination
        assert destination in ("reference_run", "twa")
    else:
        destination_array = np.array(cfg.mf_destination, float)
        assert destination_array.shape == (4, 4)
        destination = mne.transforms.Transform("meg", "head", destination_array)

    filter_chpi = cfg.mf_mc if cfg.mf_filter_chpi is None else cfg.mf_filter_chpi
    is_rest_noise = run is None and task in ("noise", "rest")
    if is_rest_noise:
        assert task is not None
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
        split=None,
    )
    bids_path_out = bids_path_in.copy().update(**bids_path_out_kwargs)

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
    # triage string-valued destinations
    if isinstance(destination, str):
        if destination == "reference_run":
            destination = raw.info["dev_head_t"]
        elif destination == "twa":
            destination = mne.read_trans(in_files.pop(f"{in_key}-twa"))
    del raw
    assert isinstance(destination, mne.transforms.Transform), destination

    # Maxwell-filter experimental data.
    apply_msg = "Preparing to apply "
    extra = list()
    if cfg.mf_st_duration:
        apply_msg += f"tSSS ({cfg.mf_st_duration} sec, corr={cfg.mf_st_correlation})"
    else:
        apply_msg += "SSS"
    if cfg.mf_mc:
        extra.append("MC")
        head_pos = mne.chpi.read_head_pos(in_files.pop(f"{in_key}-pos"))
    else:
        head_pos = None
    if cfg.mf_esss:
        extra.append("eSSS")
        extended_proj = mne.read_proj(in_files.pop("esss_basis"))
    else:
        extended_proj = ()
    if extra:
        apply_msg += " with " + "/".join(extra)
    apply_msg += " to"

    mf_kws = dict(
        calibration=in_files.pop("mf_cal_fname", None),
        cross_talk=in_files.pop("mf_ctc_fname", None),
        st_duration=cfg.mf_st_duration,
        st_correlation=cfg.mf_st_correlation,
        origin=cfg.mf_head_origin,
        coord_frame="head",
        destination=destination,
        head_pos=head_pos,
        extended_proj=extended_proj,
        int_order=cfg.mf_int_order,
        ext_order=cfg.mf_ext_order,
    )
    # If the mf_kws keys above change, we need to modify our list
    # of illegal keys in _config_import.py
    mf_kws |= cfg.mf_extra_kws

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

    if filter_chpi:
        logger.info(**gen_log_kwargs(message="Filtering cHPI"))
        mne.chpi.filter_chpi(
            raw,
            t_window=cfg.mf_mc_t_window,
            allow_line_only=(task == "noise"),
        )

    msg = (
        "Maxwell Filtering"
        f" (internal order: {mf_kws['int_order']},"
        f" external order: {mf_kws['ext_order']})"
    )
    logger.info(**gen_log_kwargs(message=msg))

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

        bids_path_ref_sss = in_files.pop("raw_ref_run_sss")
        raw_exp = mne.io.read_raw_fif(bids_path_ref_sss)
        if "grad" in raw_exp:
            if "mag" in raw_exp:
                type_sel = "meg"
            else:
                type_sel = "grad"
        else:
            type_sel = "mag"
        rank_exp = mne.compute_rank(raw_exp, rank="info")[type_sel]
        rank_noise = mne.compute_rank(raw_sss, rank="info")[type_sel]
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

    movement_annot: mne.Annotations | None = None
    extra_html: str | None = None
    if cfg.mf_mc and (
        cfg.mf_mc_rotation_velocity_limit is not None
        or cfg.mf_mc_translation_velocity_limit is not None
    ):
        movement_annot, _ = mne.preprocessing.annotate_movement(
            raw_sss,
            pos=head_pos,
            rotation_velocity_limit=cfg.mf_mc_rotation_velocity_limit,
            translation_velocity_limit=cfg.mf_mc_translation_velocity_limit,
        )
        perc_time = 100 / raw_sss.times[-1]
        extra_html_list = list()
        for kind, unit in (("translation", "m"), ("rotation", "°")):
            limit = getattr(cfg, f"mf_mc_{kind}_velocity_limit")
            if limit is None:
                continue
            desc = (f"BAD_mov_{kind[:5]}_vel",)
            tot_time = np.sum(
                movement_annot.duration[movement_annot.description == desc]
            )
            perc = perc_time * tot_time
            logger_meth = logger.warning if perc > 20 else logger.info
            msg = (
                f"{kind.capitalize()} velocity exceeded {limit} {unit}/s "
                f"limit for {tot_time:0.1f} s ({perc:0.1f}%)"
            )
            logger_meth(**gen_log_kwargs(message=msg))
            extra_html_list.append(f"<li>{msg}</li>")
        extra_html = (
            "<p>The raw data were annotated with the following movement-related bad "
            f"segment annotations:<ul>{''.join(extra_html_list)}</ul></p>"
        )
        raw_sss.set_annotations(raw_sss.annotations + movement_annot)

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
            title_prefix="Raw (maxwell filtered)",
            tags=("sss",),
            raw=raw_sss,
            extra_html=extra_html,
        )

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config_esss(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        mf_esss=config.mf_esss,
        mf_esss_reject=config.mf_esss_reject,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def get_config_maxwell_filter(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
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
        mf_ext_order=config.mf_ext_order,
        mf_mc_t_window=config.mf_mc_t_window,
        mf_mc_rotation_velocity_limit=config.mf_mc_rotation_velocity_limit,
        mf_mc_translation_velocity_limit=config.mf_mc_translation_velocity_limit,
        mf_esss=config.mf_esss,
        mf_extra_kws=config.mf_extra_kws,
        **_import_data_kwargs(config=config, subject=subject, session=session),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run maxwell_filter."""
    if not config.use_maxwell_filter:
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ss = _get_ss(config=config)
    with get_parallel_backend(config.exec_params):
        logs = list()
        # First step: compute eSSS projectors
        if config.mf_esss:
            parallel, run_func = parallel_func(
                compute_esss_proj, exec_params=config.exec_params, n_iter=len(ss)
            )
            logs += parallel(
                run_func(
                    cfg=get_config_esss(
                        config=config,
                        subject=subject,
                        session=session,
                    ),
                    exec_params=config.exec_params,
                    subject=subject,
                    session=session,
                )
                for subject, session in ss
            )

        # Second: maxwell_filter
        # We need to guarantee that the reference_run completes before the
        # noise/rest runs are processed, so we split the loops.
        for which in [("runs",), ("noise", "rest")]:
            ssrt = _get_ssrt(config=config, which=which)
            parallel, run_func = parallel_func(
                run_maxwell_filter, exec_params=config.exec_params, n_iter=len(ssrt)
            )
            logs += parallel(
                run_func(
                    cfg=get_config_maxwell_filter(
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
