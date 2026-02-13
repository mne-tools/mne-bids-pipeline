"""Group average at the source level.

Source estimates are morphed to the ``fsaverage`` brain.
"""

from types import SimpleNamespace

import mne
import numpy as np
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_sst,
    get_fs_subject,
    get_fs_subjects_dir,
    get_sessions,
    get_subjects,
    get_subjects_given_session,
    sanitize_cond_name,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _all_conditions, _get_prefix_tags, _open_report
from mne_bids_pipeline._run import _prep_out_files, failsafe_run, save_logs
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def _stc_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    condition: str,
    morphed: bool,
) -> BIDSPath:
    cond_str = sanitize_cond_name(condition)
    suffix_list = [cond_str, cfg.inverse_method, "hemi"]
    if morphed:
        suffix_list.insert(2, "morph2fsaverage")
    suffix = "+".join(suffix_list)
    del suffix_list
    return BIDSPath(
        subject=subject,
        session=session,
        task=task,
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
    session: str | None,
    task: str | None,
) -> InFilesT:
    in_files = dict()
    for condition in _all_conditions(cfg=cfg, task=task):
        in_files[f"original-{condition}"] = _stc_path(
            cfg=cfg,
            subject=subject,
            session=session,
            task=task,
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
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    out_files = dict()
    for condition in _all_conditions(cfg=cfg, task=task):
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
            task=task,
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
    session: str | None,
    task: str | None,
) -> InFilesT:
    in_files = dict()
    assert subject == "average"
    # for each session, only use subjects who actually have data for that session
    subjects = get_subjects_given_session(cfg, session)
    for condition in _all_conditions(cfg=cfg, task=task):
        for this_subject in subjects:
            in_files[f"{this_subject}-{condition}"] = _stc_path(
                cfg=cfg,
                subject=this_subject,
                session=session,
                task=task,
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
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    assert subject == "average"
    out_files = dict()
    conditions = _all_conditions(cfg=cfg, task=task)
    # for each session, only use subjects who actually have data for that session
    subjects = get_subjects_given_session(cfg, session)
    for condition in conditions:
        stc = np.array(
            [
                mne.read_source_estimate(in_files.pop(f"{this_subject}-{condition}"))
                for this_subject in subjects
            ]
        ).mean(axis=0)
        out_files[condition] = _stc_path(
            cfg=cfg,
            subject=subject,
            session=session,
            task=task,
            condition=condition,
            morphed=True,
        )
        stc.save(out_files[condition], ftype="h5", overwrite=True)

    #######################################################################
    #
    # Visualize forward solution, inverse operator, and inverse solutions.
    #
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, task=task
    ) as report:
        for condition in conditions:
            prefix, extra_tags = _get_prefix_tags(
                cfg=cfg, task=task, condition=condition
            )
            msg = f"Rendering inverse solution for {condition}"
            logger.info(**gen_log_kwargs(message=msg))
            tags: tuple[str, ...] = ("source-estimate",) + extra_tags
            if condition in cfg.conditions:
                title = f"Average (source){prefix}"
            else:  # It's a contrast of two conditions.
                title = f"Average (source) contrast{prefix}"
                tags = tags + ("contrast",)
            tags += extra_tags
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
        allow_missing_sessions=config.allow_missing_sessions,
        use_template_mri=config.use_template_mri,
        contrasts=config.contrasts,
        report_stc_n_time_points=config.report_stc_n_time_points,
        # TODO: needed because get_datatype gets called again...
        data_type=config.data_type,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    average_subj = "average"
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg, subject=average_subj))
        return

    mne.datasets.fetch_fsaverage(subjects_dir=get_fs_subjects_dir(config))
    cfg = get_config(config=config)
    exec_params = config.exec_params
    all_sessions = get_sessions(config)

    if hasattr(exec_params.overrides, "subjects"):
        msg = "Skipping, --subject is set …"
        logger.info(**gen_log_kwargs(message=msg, subject=average_subj))
        return

    logs = list()
    sst = _get_sst(config=config)
    with get_parallel_backend(exec_params):
        parallel, run_func = parallel_func(
            morph_stc, exec_params=exec_params, n_iter=len(sst)
        )
        logs += parallel(
            run_func(
                cfg=cfg,
                exec_params=exec_params,
                subject=subject,
                fs_subject=get_fs_subject(config=cfg, subject=subject, session=session),
                session=session,
                task=task,
            )
            for subject, session, task in sst
        )
    logs += [
        run_average(
            cfg=cfg,
            exec_params=exec_params,
            session=session,
            subject=average_subj,
            task=task,
        )
        for session in all_sessions
        for task in config.all_tasks
    ]
    save_logs(config=config, logs=logs)
