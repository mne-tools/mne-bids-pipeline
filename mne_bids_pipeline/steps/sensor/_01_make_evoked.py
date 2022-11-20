"""Extract evoked data for each condition."""

from types import SimpleNamespace
from typing import Optional

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype,
    get_all_contrasts, _restrict_analyze_channels,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._report import _open_report, _sanitize_cond_tag
from ..._run import failsafe_run, save_logs, _sanitize_callable


def get_input_fnames_evoked(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    fname_epochs = BIDSPath(subject=subject,
                            session=session,
                            task=cfg.task,
                            acquisition=cfg.acq,
                            run=None,
                            recording=cfg.rec,
                            space=cfg.space,
                            suffix='epo',
                            extension='.fif',
                            datatype=cfg.datatype,
                            root=cfg.deriv_root,
                            processing='clean',  # always use clean epochs
                            check=False)
    in_files = dict()
    in_files['epochs'] = fname_epochs
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_evoked,
)
def run_evoked(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    out_files = dict()
    out_files['evoked'] = in_files['epochs'].copy().update(
        suffix='ave', processing=None, check=False)

    msg = f'Input: {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg))
    msg = f'Output: {out_files["evoked"].basename}'
    logger.info(**gen_log_kwargs(message=msg))

    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    msg = 'Creating evoked data based on experimental conditions …'
    logger.info(**gen_log_kwargs(message=msg))
    all_evoked = dict()

    if isinstance(cfg.conditions, dict):
        for new_cond_name, orig_cond_name in cfg.conditions.items():
            evoked = epochs[orig_cond_name].average()
            evoked.comment = evoked.comment.replace(orig_cond_name,
                                                    new_cond_name)
            all_evoked[new_cond_name] = evoked
    else:
        for condition in cfg.conditions:
            evoked = epochs[condition].average()
            all_evoked[condition] = evoked

    if cfg.contrasts:
        msg = 'Contrasting evoked responses …'
        logger.info(**gen_log_kwargs(message=msg))

        for contrast in cfg.contrasts:
            evoked_list = [epochs[x].average() for x in contrast["conditions"]]
            evoked_diff = mne.combine_evoked(evoked_list,
                                             weights=contrast["weights"])
            all_evoked[contrast["name"]] = evoked_diff

    evokeds = list(all_evoked.values())
    for evoked in evokeds:
        evoked.nave = int(round(evoked.nave))  # avoid a warning
    mne.write_evokeds(out_files['evoked'], evokeds, overwrite=True)

    # Report
    if evokeds:
        msg = (f'Adding {len(evokeds)} evoked signals and contrasts to the '
               f'report.')
    else:
        msg = 'No evoked conditions or contrasts found.'
    logger.info(
        **gen_log_kwargs(message=msg)
    )
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        for condition, evoked in all_evoked.items():
            _restrict_analyze_channels(evoked, cfg)

            tags = ('evoked', _sanitize_cond_tag(condition))
            if condition in cfg.conditions:
                title = f'Condition: {condition}'
            else:  # It's a contrast of two conditions.
                title = f'Contrast: {condition}'
                tags = tags + ('contrast',)

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

    return out_files


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.deriv_root,
        conditions=config.conditions,
        contrasts=get_all_contrasts(config),
        proc=config.proc,
        noise_cov=_sanitize_callable(config.noise_cov),
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        report_evoked_n_time_points=config.report_evoked_n_time_points,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run evoked."""
    if config.task_is_rest:
        msg = '    … skipping: for resting-state task.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_evoked, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
