"""Extract evoked data for each condition."""

from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_sst,
    _get_task_conditions_dict,
    _get_task_contrasts,
    _pl,
    _restrict_analyze_channels,
    get_eeg_reference,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import (
    _all_conditions,
    _get_prefix_tags,
    _open_report,
)
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_evoked(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> InFilesT:
    fname_epochs = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix="epo",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        processing="clean",  # always use clean epochs
        check=False,
    )
    in_files = dict()
    in_files["epochs"] = fname_epochs
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_evoked,
)
def run_evoked(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    out_files = dict()
    out_files["evoked"] = (
        in_files["epochs"]
        .copy()
        .update(
            suffix="ave",
            processing=None,
            check=False,
            split=None,
        )
    )

    msg = f"Input: {in_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['evoked'].basename}"
    logger.info(**gen_log_kwargs(message=msg))

    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    msg = "Creating evoked data based on experimental conditions …"
    logger.info(**gen_log_kwargs(message=msg))
    all_evoked = dict()
    for new_cond_name, orig_cond_name in cfg.conditions.items():
        evoked = epochs[orig_cond_name].average()
        evoked.comment = evoked.comment.replace(orig_cond_name, new_cond_name)
        all_evoked[new_cond_name] = evoked

    if cfg.contrasts:
        msg = "Contrasting evoked responses …"
        logger.info(**gen_log_kwargs(message=msg))

        for contrast in cfg.contrasts:
            evoked_list = [epochs[x].average() for x in contrast["conditions"]]
            evoked_diff = mne.combine_evoked(evoked_list, weights=contrast["weights"])
            all_evoked[contrast["name"]] = evoked_diff

    evokeds = list(all_evoked.values())
    for evoked in evokeds:
        evoked.nave = int(round(evoked.nave))  # avoid a warning
    mne.write_evokeds(out_files["evoked"], evokeds, overwrite=True)

    # Report
    if evokeds:
        n_contrasts = len(cfg.contrasts)
        n_signals = len(evokeds) - n_contrasts
        msg = (
            f"Adding {n_signals} evoked response{_pl(n_signals)} and "
            f"{n_contrasts} contrast{_pl(n_contrasts)} to the report."
        )
    else:
        msg = "No evoked conditions or contrasts found."
    logger.info(**gen_log_kwargs(message=msg))
    all_conditions = _all_conditions(cfg=cfg, task=task)
    assert list(all_conditions) == list(all_evoked)  # otherwise we have a bug
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, task=task
    ) as report:
        for condition, evoked in all_evoked.items():
            _restrict_analyze_channels(evoked, cfg)

            prefix, extra_tags = _get_prefix_tags(
                cfg=cfg, task=task, condition=condition
            )
            tags: tuple[str, ...] = ("evoked",) + extra_tags
            if condition in cfg.conditions:
                title = f"Condition{prefix}"
            else:  # It's a contrast of two conditions.
                title = f"Contrast{prefix}"
                tags = tags + ("contrast",)

            report.add_evokeds(
                evokeds=evoked,
                titles=title,
                n_time_points=cfg.report_evoked_n_time_points,
                tags=tags,
                replace=True,
                n_jobs=1,  # don't auto parallelize
            )

    # Interaction
    if exec_params.interactive:
        for evoked in evokeds:
            evoked.plot()

        # What's next needs channel locations
        # ts_args = dict(gfp=True, time_unit='s')
        # topomap_args = dict(time_unit='s')

        # for condition, evoked in zip(config.conditions, evokeds):
        #     evoked.plot_joint(title=condition, ts_args=ts_args,
        #                       topomap_args=topomap_args)
    assert len(in_files) == 0, in_files.keys()

    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    task: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        conditions=_get_task_conditions_dict(conditions=config.conditions, task=task),
        contrasts=_get_task_contrasts(contrasts=config.contrasts, task=task),
        analyze_channels=config.analyze_channels,
        eeg_reference=get_eeg_reference(config),
        ch_types=config.ch_types,
        report_evoked_n_time_points=config.report_evoked_n_time_points,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run evoked."""
    if config.task_is_rest:
        msg = "Skipping, resting-state task …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    sst = _get_sst(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_evoked, exec_params=config.exec_params, n_iter=len(sst)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
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
