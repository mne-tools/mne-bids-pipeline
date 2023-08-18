"""Inverse solution.

Compute and apply an inverse solution for each evoked data set.
"""

from types import SimpleNamespace
from typing import Optional

import mne
from mne.minimum_norm import (
    make_inverse_operator,
    apply_inverse,
    write_inverse_operator,
)
from mne_bids import BIDSPath

from ..._config_utils import (
    get_noise_cov_bids_path,
    get_subjects,
    sanitize_cond_name,
    get_sessions,
    get_fs_subjects_dir,
    get_fs_subject,
    _bids_kwargs,
)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _open_report, _sanitize_cond_tag, _all_conditions
from ..._run import failsafe_run, save_logs, _sanitize_callable, _prep_out_files


def get_input_fnames_inverse(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
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
    in_files["info"] = bids_path.copy().update(**cfg.source_info_path_update)
    in_files["forward"] = bids_path.copy().update(suffix="fwd")
    if cfg.noise_cov != "ad-hoc":
        in_files["cov"] = get_noise_cov_bids_path(
            cfg=cfg, subject=subject, session=session
        )
    if "evoked" in cfg.inverse_targets:
        in_files["evoked"] = bids_path.copy().update(suffix="ave")
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_inverse,
)
def run_inverse(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    # TODO: Eventually we should maybe loop over ch_types, e.g., to create
    # MEG, EEG, and MEG+EEG inverses and STCs
    msg = "Computing inverse solutions"
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
    inverse_operator = make_inverse_operator(
        info, forward, cov, loose=cfg.loose, depth=cfg.depth, rank="info"
    )
    write_inverse_operator(out_files["inverse"], inverse_operator, overwrite=True)

    # Apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr**2
    conditions = _all_conditions(cfg=cfg)
    method = cfg.inverse_method
    if "evoked" in in_files:
        fname_ave = in_files.pop("evoked")
        evokeds = mne.read_evokeds(fname_ave)

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
            cfg=cfg, exec_params=exec_params, subject=subject, session=session
        ) as report:
            msg = "Adding inverse information to report"
            logger.info(**gen_log_kwargs(message=msg))
            for condition in conditions:
                msg = f"Rendering inverse solution for {condition}"
                logger.info(**gen_log_kwargs(message=msg))
                tags = ("source-estimate", _sanitize_cond_tag(condition))
                if condition not in cfg.conditions:
                    tags = tags + ("contrast",)
                report.add_stc(
                    stc=out_files[condition],
                    title=f"Source: {condition}",
                    subject=cfg.fs_subject,
                    subjects_dir=cfg.fs_subjects_dir,
                    n_time_points=cfg.report_stc_n_time_points,
                    tags=tags,
                    replace=True,
                )

    assert len(in_files) == 0, in_files
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        source_info_path_update=config.source_info_path_update,
        inverse_targets=config.inverse_targets,
        ch_types=config.ch_types,
        conditions=config.conditions,
        contrasts=config.contrasts,
        loose=config.loose,
        depth=config.depth,
        inverse_method=config.inverse_method,
        noise_cov=_sanitize_callable(config.noise_cov),
        report_stc_n_time_points=config.report_stc_n_time_points,
        fs_subject=get_fs_subject(config=config, subject=subject),
        fs_subjects_dir=get_fs_subjects_dir(config),
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run inv."""
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(run_inverse, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
