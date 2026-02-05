"""Inverse solution.

Compute and apply an inverse solution for each evoked data set.
"""

from types import SimpleNamespace

import mne
from mne.minimum_norm import (
    apply_inverse,
    make_inverse_operator,
    write_inverse_operator,
)
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_ss,
    _get_sst,
    _get_task_conditions_dict,
    _get_task_contrasts,
    get_fs_subject,
    get_fs_subjects_dir,
    get_noise_cov_bids_path,
    sanitize_cond_name,
)
from mne_bids_pipeline._io import _read_json
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import (
    _all_conditions,
    _get_prefix_tags,
    _open_report,
)
from mne_bids_pipeline._run import (
    _prep_out_files,
    _sanitize_callable,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_make_inverse(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> InFilesT:
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=None,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files = dict()
    # make sure the info matches the data from which the noise cov
    # is computed to avoid rank-mismatch
    if cfg.source_info_path_update is None:
        if cfg.noise_cov in ("rest", "noise"):
            source_info_path_update = dict(
                processing="clean", suffix="raw", task=cfg.noise_cov
            )
        else:
            source_info_path_update = dict(suffix="ave", task=cfg.all_tasks[0])
            # XXX is this the right solution also for noise_cov = 'ad-hoc'?
    else:
        source_info_path_update = cfg.source_info_path_update
    in_files["info"] = bids_path.copy().update(**source_info_path_update)
    in_files["forward"] = bids_path.copy().update(suffix="fwd")
    cov_path = get_noise_cov_bids_path(cfg=cfg, subject=subject, session=session)
    if cfg.noise_cov != "ad-hoc":
        in_files["cov"] = cov_path
    in_files["rank"] = cov_path.copy().update(suffix="rank", extension=".json")
    return in_files


def get_input_fnames_apply_inverse_data(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> InFilesT:
    inverse_path = BIDSPath(
        subject=subject,
        session=session,
        task=None,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        suffix="inv",
        check=False,
    )
    in_files = dict()
    in_files["inverse"] = inverse_path
    in_files["evoked"] = inverse_path.copy().update(suffix="ave", task=task)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_make_inverse,
)
def make_inverse(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    # TODO: Eventually we should maybe loop over ch_types, e.g., to create
    # MEG, EEG, and MEG+EEG inverses and STCs
    msg = "Computing inverse solution"
    logger.info(**gen_log_kwargs(message=msg))
    fname_fwd = in_files.pop("forward")
    out_files = dict()
    out_files["inverse"] = fname_fwd.copy().update(suffix="inv")

    info = mne.io.read_info(in_files.pop("info"))

    if cfg.noise_cov == "ad-hoc":
        cov = mne.make_ad_hoc_cov(info)
    else:
        cov = mne.read_cov(in_files.pop("cov"))

    forward = mne.read_forward_solution(fname_fwd)
    del fname_fwd
    rank = _read_json(in_files.pop("rank"))
    inverse_operator = make_inverse_operator(
        info, forward, cov, loose=cfg.loose, depth=cfg.depth, rank=rank
    )
    write_inverse_operator(out_files["inverse"], inverse_operator, overwrite=True)
    assert len(in_files) == 0, in_files
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


@failsafe_run(
    get_input_fnames=get_input_fnames_apply_inverse_data,
)
def apply_inverse_data(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    # Apply inverse
    msg = "Applying inverse to data"
    logger.info(**gen_log_kwargs(message=msg))

    out_files = dict()
    snr = 3.0
    lambda2 = 1.0 / snr**2
    conditions = _all_conditions(cfg=cfg, task=task)
    method = cfg.inverse_method
    fname_ave = in_files.pop("evoked")
    fname_inv = in_files.pop("inverse")
    evokeds = mne.read_evokeds(fname_ave)
    assert isinstance(evokeds, list)
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    for condition, evoked in zip(conditions, evokeds):
        suffix = f"{sanitize_cond_name(condition)}+{method}+hemi"
        out_files[condition] = fname_ave.copy().update(
            suffix=suffix,
            extension=".h5",
        )

        if "eeg" in cfg.ch_types:
            evoked.set_eeg_reference("average", projection=True)

        stc = apply_inverse(
            evoked=evoked,
            inverse_operator=inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=None,
        )
        stc.save(out_files[condition], ftype="h5", overwrite=True)

    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, task=task
    ) as report:
        prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=task, condition=condition)
        for condition in conditions:
            msg = f"Rendering inverse solution for {condition=}"
            logger.info(**gen_log_kwargs(message=msg))
            tags = ("source-estimate",) + extra_tags
            if condition not in cfg.conditions:
                tags = tags + ("contrast",)
            tags += extra_tags
            report.add_stc(
                stc=out_files[condition],
                title=f"Source{prefix}",
                subject=cfg.fs_subject,
                subjects_dir=cfg.fs_subjects_dir,
                n_time_points=cfg.report_stc_n_time_points,
                tags=tags,
                replace=True,
            )

    assert len(in_files) == 0, in_files
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config_make_inverse(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        source_info_path_update=config.source_info_path_update,
        loose=config.loose,
        depth=config.depth,
        inverse_method=config.inverse_method,
        noise_cov=_sanitize_callable(config.noise_cov),
        **_bids_kwargs(config=config),
    )
    return cfg


def get_config_apply_inverse_data(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
    task: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        ch_types=config.ch_types,
        inverse_method=config.inverse_method,
        conditions=_get_task_conditions_dict(conditions=config.conditions, task=task),
        contrasts=_get_task_contrasts(contrasts=config.contrasts, task=task),
        report_stc_n_time_points=config.report_stc_n_time_points,
        fs_subject=get_fs_subject(config=config, subject=subject, session=session),
        fs_subjects_dir=get_fs_subjects_dir(config),
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run inv."""
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    ss = _get_ss(config=config)
    sst = _get_sst(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            make_inverse, exec_params=config.exec_params, n_iter=len(ss)
        )
        logs = parallel(
            run_func(
                cfg=get_config_make_inverse(
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
        if "evoked" in config.inverse_targets:
            parallel, run_func = parallel_func(
                apply_inverse_data, exec_params=config.exec_params, n_iter=len(sst)
            )
            logs += parallel(
                run_func(
                    cfg=get_config_apply_inverse_data(
                        config=config,
                        subject=subject,
                        session=session,
                        task=task,
                    ),
                    exec_params=config.exec_params,
                    subject=subject,
                    session=session,
                    task=task,
                )
                for subject, session, task in sst
            )
    save_logs(config=config, logs=logs)
