"""Extract epochs.

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id).
Finally the epochs are saved to disk. For the moment, no rejection is applied.
To save space, the epoch data can be decimated.
"""

import inspect
from types import SimpleNamespace
from typing import Any

import mne
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_runs_sst,
    _get_sst,
    _get_task_float,
    get_eeg_reference,
)
from mne_bids_pipeline._import_data import annotations_to_events, make_epochs
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _get_prefix_tags, _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _sanitize_callable,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, IntArrayT, OutFilesT


def get_input_fnames_epochs(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None = None,
) -> InFilesT:
    """Get paths of files required by filter_data function."""
    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        processing=cfg.processing,
    ).update(suffix="raw", check=False)

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.
    in_files = dict()
    for run in cfg.runs_for_task:
        key = f"raw_run-{run}"
        in_files[key] = bids_path.copy().update(run=run, task=task)
        _update_for_splits(in_files, key, single=True)
    if cfg.use_maxwell_filter and cfg.noise_cov == "rest":
        in_files["raw_rest"] = bids_path.copy().update(task="rest", check=False)
        _update_for_splits(in_files, "raw_rest", single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_epochs,
)
def run_epochs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None = None,
    in_files: InFilesT,
) -> OutFilesT:
    """Extract epochs for one subject."""
    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs_for_task]
    bids_path_in = raw_fnames[0].copy().update(processing=None, run=None, split=None)

    # Generate a unique event name -> event code mapping that can be used
    # across all runs.
    if not cfg.task_is_rest:
        event_name_to_code_map = annotations_to_events(raw_paths=raw_fnames)

    # Store the rank & corresponding info of the run with the smallest rank.
    # We'll later manually inject this info into concatenated epochs.
    # This is ONLY relevant for Maxwell-filtered MEG data and ensures that
    # we later don't assume a rank that is too large when whitening the data
    # or performing the inverse modeling.
    smallest_rank = None
    smallest_rank_info = None

    # Now, generate epochs from each individual run.
    for idx, (run, raw_fname) in enumerate(zip(cfg.runs_for_task, raw_fnames)):
        msg = f"Loading filtered raw data from {raw_fname.basename}"
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(raw_fname).load_data()

        # Only keep the subset of the mapping that applies to the current run
        if cfg.task_is_rest:
            event_id = None  # make_epochs takes care of it.
        else:
            event_id = event_name_to_code_map.copy()
            for event_name in event_id.copy().keys():
                if event_name not in raw.annotations.description:
                    del event_id[event_name]

        msg = "Creating task-related epochs …"
        logger.info(**gen_log_kwargs(message=msg))
        epochs = make_epochs(
            subject=subject,
            session=session,
            task=task,
            raw=raw,
            event_id=event_id,
            conditions=cfg.conditions,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            custom_metadata=cfg.epochs_custom_metadata,
            metadata_tmin=cfg.epochs_metadata_tmin,
            metadata_tmax=cfg.epochs_metadata_tmax,
            metadata_keep_first=cfg.epochs_metadata_keep_first,
            metadata_keep_last=cfg.epochs_metadata_keep_last,
            metadata_query=cfg.epochs_metadata_query,
            event_repeated=cfg.event_repeated,
            epochs_decim=cfg.epochs_decim,
            task_is_rest=cfg.task_is_rest,
            rest_epochs_duration=cfg.rest_epochs_duration,
            rest_epochs_overlap=cfg.rest_epochs_overlap,
        )

        epochs.load_data()  # Remove reference to raw
        del raw  # free memory

        if idx == 0:
            epochs_all_runs = epochs
        else:
            epochs_all_runs = mne.concatenate_epochs(
                [epochs_all_runs, epochs], on_mismatch="warn"
            )

        if cfg.use_maxwell_filter:
            # Keep track of the info corresponding to the run with the smallest
            # data rank.
            if "grad" in epochs:
                if "mag" in epochs:
                    type_sel = "meg"
                else:
                    type_sel = "grad"
            else:
                type_sel = "mag"
            new_rank = mne.compute_rank(epochs, rank="info")[type_sel]
            if (smallest_rank is None) or (new_rank < smallest_rank):
                smallest_rank = new_rank
                smallest_rank_info = epochs.info.copy()

        del epochs, run

    # Clean up namespace
    epochs = epochs_all_runs
    del epochs_all_runs

    if cfg.use_maxwell_filter and cfg.noise_cov == "rest":
        raw_rest_filt = mne.io.read_raw(in_files.pop("raw_rest"))
        rank_rest = mne.compute_rank(raw_rest_filt, rank="info")[type_sel]
        if rank_rest < smallest_rank:
            msg = (
                f"The MEG rank of the resting state data ({rank_rest}) is "
                f'smaller than the smallest MEG rank of the "{task}" '
                f'epochs ({smallest_rank}). Replacing part of the  "info" '
                f'object of the concatenated "{task}" epochs with '
                f"information from the resting-state run."
            )
            logger.warning(**gen_log_kwargs(message=msg, run="rest"))
            smallest_rank = rank_rest
            smallest_rank_info = raw_rest_filt.info.copy()

        del raw_rest_filt

    if cfg.use_maxwell_filter:
        # Inject the Maxwell filter info corresponding to the run with the
        # smallest data rank, so when deducing the rank of the data from the
        # info, it will be the smallest rank of any bit of data we're
        # processing. This is to prevent issues during the source estimation
        # step.
        assert smallest_rank_info is not None
        assert epochs.info["ch_names"] == smallest_rank_info["ch_names"]
        with epochs.info._unlock():
            epochs.info["proc_history"] = smallest_rank_info["proc_history"]
            rank_epochs_new = mne.compute_rank(epochs, rank="info")[type_sel]
            msg = f'The MEG rank of the "{task}" epochs is now: {rank_epochs_new}'
            logger.warning(**gen_log_kwargs(message=msg))

    # Set an EEG reference
    if "eeg" in cfg.ch_types:
        projection = True if cfg.eeg_reference == "average" else False
        epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

    n_epochs_before_metadata_query = len(epochs.drop_log)

    msg = (
        f"Created {n_epochs_before_metadata_query} epochs with time "
        f"interval: {epochs.tmin} – {epochs.tmax} sec."
    )
    logger.info(**gen_log_kwargs(message=msg))
    msg = (
        f"Selected {len(epochs)} epochs via metadata query: {cfg.epochs_metadata_query}"
    )
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Writing {len(epochs)} epochs to disk."
    logger.info(**gen_log_kwargs(message=msg))
    out_files = dict()
    out_files["epochs"] = bids_path_in.copy().update(
        suffix="epo",
        processing=None,
        check=False,
        split=None,
    )
    epochs.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, task=task
    ) as report:
        prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=task)
        if not cfg.task_is_rest:
            msg = "Adding events plot to report."
            logger.info(**gen_log_kwargs(message=msg))
            events, event_id, sfreq, first_samp = _get_events(
                cfg=cfg, subject=subject, session=session, task=task
            )
            report.add_events(
                events=events,
                event_id=event_id,
                sfreq=sfreq,
                first_samp=first_samp,
                title=f"Events{prefix}",
                tags=("events",) + extra_tags,
                # caption='Events in filtered continuous data',  # TODO upstr
                replace=True,
            )
        msg = "Adding uncleaned epochs to report."
        logger.info(**gen_log_kwargs(message=msg))
        # Add PSD plots for 30s of data or all epochs if we have less available
        psd = True if len(epochs) * (epochs.tmax - epochs.tmin) < 30 else 30.0
        report.add_epochs(
            epochs=epochs,
            title=f"Epochs (before cleaning){prefix}",
            psd=psd,
            drop_log_ignore=(),
            replace=True,
            tags=("epochs",) + extra_tags,
            **_add_epochs_image_kwargs(cfg),
        )

    # Interactive
    if exec_params.interactive:
        epochs.plot()
        epochs.plot_image(combine="gfp", sigma=2.0, cmap="YlGnBu_r")
    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def _add_epochs_image_kwargs(cfg: SimpleNamespace) -> dict[str, dict[str, Any]]:
    arg_spec = inspect.getfullargspec(mne.Report.add_epochs)
    kwargs = dict()
    if cfg.report_add_epochs_image_kwargs and "image_kwargs" in arg_spec.kwonlyargs:
        kwargs["image_kwargs"] = cfg.report_add_epochs_image_kwargs
    return kwargs


# TODO: ideally we wouldn't need this anymore and could refactor the code above
def _get_events(
    *, cfg: SimpleNamespace, subject: str, session: str | None, task: str | None = None
) -> tuple[IntArrayT, dict[str, int], float, int]:
    raws_filt = []
    raw_fname = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        processing=cfg.processing,
        suffix="raw",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )

    for run in cfg.runs_for_task:
        this_raw_fname = raw_fname.copy().update(run=run)
        this_raw_fname = _update_for_splits(this_raw_fname, None, single=True)
        raw_filt = mne.io.read_raw_fif(this_raw_fname)
        raws_filt.append(raw_filt)
        del this_raw_fname

    # Concatenate the filtered raws and extract the events.
    raw_filt_concat = mne.concatenate_raws(raws_filt, on_mismatch="warn")
    events, event_id = mne.events_from_annotations(raw=raw_filt_concat)
    return (events, event_id, raw_filt_concat.info["sfreq"], raw_filt_concat.first_samp)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        use_maxwell_filter=config.use_maxwell_filter,
        task_is_rest=config.task_is_rest,
        conditions=config.conditions,
        epochs_tmin=_get_task_float(config.epochs_tmin, task=task),
        epochs_tmax=_get_task_float(config.epochs_tmax, task=task),
        epochs_custom_metadata=config.epochs_custom_metadata,
        epochs_metadata_tmin=config.epochs_metadata_tmin,
        epochs_metadata_tmax=config.epochs_metadata_tmax,
        epochs_metadata_keep_first=config.epochs_metadata_keep_first,
        epochs_metadata_keep_last=config.epochs_metadata_keep_last,
        epochs_metadata_query=config.epochs_metadata_query,
        event_repeated=config.event_repeated,
        epochs_decim=config.epochs_decim,
        report_add_epochs_image_kwargs=config.report_add_epochs_image_kwargs,
        ch_types=config.ch_types,
        noise_cov=_sanitize_callable(config.noise_cov),
        eeg_reference=get_eeg_reference(config),
        rest_epochs_duration=config.rest_epochs_duration,
        rest_epochs_overlap=config.rest_epochs_overlap,
        _epochs_split_size=config._epochs_split_size,
        runs_for_task=_get_runs_sst(
            config=config, subject=subject, session=session, task=task
        ),
        processing="filt" if config.regress_artifact is None else "regress",
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run epochs."""
    sst = _get_sst(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_epochs, exec_params=config.exec_params, n_iter=len(sst)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
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
