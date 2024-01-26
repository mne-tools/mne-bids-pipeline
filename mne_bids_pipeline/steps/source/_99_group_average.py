"""Group average at the source level.

Source estimates are morphed to the ``fsaverage`` brain.
"""

from types import SimpleNamespace
from typing import Optional

import mne
import numpy as np
from mne_bids import BIDSPath

from ..._config_utils import (
    _bids_kwargs,
    get_fs_subject,
    get_fs_subjects_dir,
    get_sessions,
    get_subjects,
    sanitize_cond_name,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _all_conditions, _open_report
from ..._run import _prep_out_files, failsafe_run, save_logs


def _stc_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    condition: str,
    morphed: bool,
) -> BIDSPath:
    cond_str = sanitize_cond_name(condition)
    suffix = [cond_str, cfg.inverse_method, "hemi"]
    if morphed:
        suffix.insert(2, "morph2fsaverage")
    suffix = "+".join(suffix)
    return BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        suffix=suffix,
        extension=".h5",
        check=False,
    )


def get_input_fnames_morph_stc(
    *,
    cfg: SimpleNamespace,
    subject: str,
    fs_subject: str,
    session: Optional[str],
) -> dict:
    in_files = dict()
    for condition in _all_conditions(cfg=cfg):
        in_files[f"original-{condition}"] = _stc_path(
            cfg=cfg,
            subject=subject,
            session=session,
            condition=condition,
            morphed=False,
        )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_morph_stc,
)
def morph_stc(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    fs_subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    out_files = dict()
    for condition in _all_conditions(cfg=cfg):
        fname_stc = in_files.pop(f"original-{condition}")
        stc = mne.read_source_estimate(fname_stc)
        morph = mne.compute_source_morph(
            stc,
            subject_from=fs_subject,
            subject_to="fsaverage",
            subjects_dir=cfg.fs_subjects_dir,
        )
        stc_fsaverage = morph.apply(stc)
        key = f"morphed-{condition}"
        out_files[key] = _stc_path(
            cfg=cfg,
            subject=subject,
            session=session,
            condition=condition,
            morphed=True,
        )
        stc_fsaverage.save(out_files[key], ftype="h5", overwrite=True)

    assert len(in_files) == 0, in_files
    return _prep_out_files(out_files=out_files, exec_params=exec_params)


def get_input_fnames_run_average(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    in_files = dict()
    assert subject == "average"
    for condition in _all_conditions(cfg=cfg):
        for this_subject in cfg.subjects:
            in_files[f"{this_subject}-{condition}"] = _stc_path(
                cfg=cfg,
                subject=this_subject,
                session=session,
                condition=condition,
                morphed=True,
            )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_average,
)
def run_average(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
):
    assert subject == "average"
    out_files = dict()
    conditions = _all_conditions(cfg=cfg)
    for condition in conditions:
        stc = np.array(
            [
                mne.read_source_estimate(in_files.pop(f"{this_subject}-{condition}"))
                for this_subject in cfg.subjects
            ]
        ).mean(axis=0)
        out_files[condition] = _stc_path(
            cfg=cfg,
            subject=subject,
            session=session,
            condition=condition,
            morphed=True,
        )
        stc.save(out_files[condition], ftype="h5", overwrite=True)

    #######################################################################
    #
    # Visualize forward solution, inverse operator, and inverse solutions.
    #
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        for condition in conditions:
            msg = f"Rendering inverse solution for {condition}"
            logger.info(**gen_log_kwargs(message=msg))
            cond_str = sanitize_cond_name(condition)
            tags = ("source-estimate", cond_str)
            if condition in cfg.conditions:
                title = f"Average (source): {condition}"
            else:  # It's a contrast of two conditions.
                title = f"Average (source) contrast: {condition}"
                tags = tags + ("contrast",)
            report.add_stc(
                stc=out_files[condition],
                title=title,
                subject="fsaverage",
                subjects_dir=cfg.fs_subjects_dir,
                n_time_points=cfg.report_stc_n_time_points,
                tags=tags,
                replace=True,
            )
    assert len(in_files) == 0, in_files
    return _prep_out_files(out_files=out_files, exec_params=exec_params)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task_is_rest=config.task_is_rest,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        fs_subjects_dir=get_fs_subjects_dir(config),
        subjects_dir=get_fs_subjects_dir(config),
        ch_types=config.ch_types,
        subjects=get_subjects(config=config),
        exclude_subjects=config.exclude_subjects,
        sessions=get_sessions(config),
        use_template_mri=config.use_template_mri,
        contrasts=config.contrasts,
        report_stc_n_time_points=config.report_stc_n_time_points,
        # TODO: needed because get_datatype gets called again...
        data_type=config.data_type,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    mne.datasets.fetch_fsaverage(subjects_dir=get_fs_subjects_dir(config))
    cfg = get_config(config=config)
    exec_params = config.exec_params
    subjects = get_subjects(config)
    sessions = get_sessions(config)

    logs = list()
    with get_parallel_backend(exec_params):
        parallel, run_func = parallel_func(morph_stc, exec_params=exec_params)
        logs += parallel(
            run_func(
                cfg=cfg,
                exec_params=exec_params,
                subject=subject,
                fs_subject=get_fs_subject(config=cfg, subject=subject),
                session=session,
            )
            for subject in subjects
            for session in sessions
        )
    logs += [
        run_average(
            cfg=cfg,
            exec_params=exec_params,
            session=session,
            subject="average",
        )
        for session in sessions
    ]
    save_logs(config=config, logs=logs)
