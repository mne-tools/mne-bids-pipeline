from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Literal

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_bids_path_from_fname, read_raw_bids

from ._config_utils import (
    _bids_kwargs,
    _pl,
    get_datatype,
    get_eog_channels,
    get_mf_reference_run,
    get_runs,
    get_task,
)
from ._io import _read_json
from ._logging import gen_log_kwargs, logger
from ._run import _update_for_splits
from .typing import InFilesT, PathLike, RunKindT, RunTypeT


def make_epochs(
    *,
    task: str,
    subject: str,
    session: str | None,
    raw: mne.io.BaseRaw,
    event_id: dict[str, int] | Literal["auto"] | None,
    conditions: Iterable[str] | dict[str, str],
    tmin: float,
    tmax: float,
    custom_metadata: pd.DataFrame | dict[str, Any] | None,
    metadata_tmin: float | None,
    metadata_tmax: float | None,
    metadata_keep_first: Iterable[str] | None,
    metadata_keep_last: Iterable[str] | None,
    metadata_query: str | None,
    event_repeated: Literal["error", "drop", "merge"],
    epochs_decim: int,
    task_is_rest: bool,
    rest_epochs_duration: float,
    rest_epochs_overlap: float,
) -> mne.Epochs:
    """Generate Epochs from raw data.

    - Only events corresponding to `conditions` will be used to create epochs.
    - Metadata queries to subset epochs will be performed.

    - No EEG reference will be set and no projectors will be applied.
    - No rejection thresholds will be applied.
    - No baseline-correction will be performed.
    """
    if task_is_rest:
        stop = raw.times[-1] - rest_epochs_duration
        assert tmin == 0.0, "epochs_tmin must be 0 for rest"
        assert rest_epochs_overlap is not None, "epochs_overlap cannot be None for rest"
        events = mne.make_fixed_length_events(
            raw,
            id=3000,
            start=0,
            duration=rest_epochs_duration,
            overlap=rest_epochs_overlap,
            stop=stop,
        )
        event_id = dict(rest=3000)
        metadata = None
    else:  # Events for task runs
        if event_id is None:
            event_id = "auto"

        events, event_id = mne.events_from_annotations(raw, event_id=event_id)

        # Construct metadata
        #
        # We only keep conditions that will be analyzed.
        if isinstance(conditions, dict):
            conditions = list(conditions.keys())
        else:
            conditions = list(conditions)  # Ensure we have a list

        # Handle grouped / hierarchical event names.
        row_event_names = mne.event.match_event_names(
            event_names=event_id, keys=conditions
        )

        if metadata_tmin is None:
            metadata_tmin = tmin
        if metadata_tmax is None:
            metadata_tmax = tmax

        # The returned `events` and `event_id` will only contain
        # the events from `row_event_names` – which is basically equivalent to
        # what the user requested via `config.conditions` (only with potential
        # nested event names expanded, e.g. `visual` might now be
        # `visual/left` and `visual/right`)
        metadata, events, event_id = mne.epochs.make_metadata(
            row_events=row_event_names,
            events=events,
            event_id=event_id,
            tmin=metadata_tmin,
            tmax=metadata_tmax,
            keep_first=metadata_keep_first,
            keep_last=metadata_keep_last,
            sfreq=raw.info["sfreq"],
        )

        # If custom_metadata is provided, merge it with the generated metadata
        if custom_metadata is not None:
            if isinstance(
                custom_metadata, dict
            ):  # parse custom_metadata['sub-x']['ses-y']['task-z']
                custom_dict = custom_metadata
                for _ in range(3):  # loop to allow for mis-ordered keys
                    if (
                        isinstance(custom_dict, dict)
                        and "subj-" + subject in custom_dict
                    ):
                        custom_dict = custom_dict["subj-" + subject]
                    if (
                        isinstance(custom_dict, dict)
                        and session is not None
                        and "ses-" + session in custom_dict
                    ):
                        custom_dict = custom_dict["ses-" + session]
                    if isinstance(custom_dict, dict) and "task-" + task in custom_dict:
                        custom_dict = custom_dict["task-" + task]
                    if isinstance(custom_dict, pd.DataFrame):
                        custom_df = custom_dict
                        break
                if not isinstance(custom_dict, pd.DataFrame):
                    msg = (
                        f"Custom metadata not found for subject {subject} / "
                        f"session {session} / task {task}.\n"
                    )
                    raise ValueError(msg)
            elif isinstance(custom_metadata, pd.DataFrame):  # parse DataFrame
                custom_df = custom_metadata
            else:
                msg = (
                    f"Custom metadata not found for subject {subject} / "
                    f"session {session} / task {task}.\n"
                )
                raise ValueError(msg)

            # Check if the custom metadata DataFrame has the same number of rows
            if len(metadata) != len(custom_df):
                msg = (
                    f"Event metadata has {len(metadata)} rows, but custom "
                    f"metadata has {len(custom_df)} rows. Cannot safely join."
                )
                raise ValueError(msg)

            # Merge the event and custom DataFrames
            metadata = metadata.join(custom_df, how="right")
            # Logging   # Logging
            msg = "Including custom metadata in epochs."
            logger.info(**gen_log_kwargs(message=msg))

    # Epoch the data
    # Do not reject based on peak-to-peak or flatness thresholds at this stage
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        baseline=None,
        preload=False,
        decim=epochs_decim,
        metadata=metadata,
        event_repeated=event_repeated,
        reject=None,
    )

    # Now, select a subset of epochs based on metadata.
    # All epochs that are omitted by the query will get a corresponding
    # entry in epochs.drop_log, allowing us to keep track of how many (and
    # which) epochs got omitted. We're first generating an index which we can
    # then pass to epochs.drop(); this allows us to specify a custom drop
    # reason.
    if metadata_query is not None:
        import pandas.core

        assert epochs.metadata is not None

        try:
            idx_keep = epochs.metadata.eval(metadata_query, engine="python")
        except pandas.core.computation.ops.UndefinedVariableError:
            msg = f"Metadata query failed to select any columns: {metadata_query}"
            logger.warning(**gen_log_kwargs(message=msg))
            return epochs

        idx_drop = epochs.metadata.index[~idx_keep]
        epochs.drop(indices=idx_drop, reason="metadata query", verbose=False)
        del idx_keep, idx_drop

    return epochs


def annotations_to_events(*, raw_paths: list[PathLike | BIDSPath]) -> dict[str, int]:
    """Generate a unique event name -> event code mapping.

    The mapping can that can be used across all passed raws.
    """
    event_names: list[str] = []
    for raw_fname in raw_paths:
        raw = mne.io.read_raw_fif(raw_fname)
        _, event_id = mne.events_from_annotations(raw=raw)
        for event_name in event_id.keys():
            if event_name not in event_names:
                event_names.append(event_name)

    event_names = sorted(event_names)
    event_name_to_code_map = {
        name: code for code, name in enumerate(event_names, start=1)
    }

    return event_name_to_code_map


def _rename_events_func(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str | None,
) -> None:
    """Rename events (actually, annotations descriptions) in ``raw``.

    Modifies ``raw`` in-place.
    """
    if not cfg.rename_events:
        return

    # Check if the user requested to rename events that don't exist.
    # We don't want this to go unnoticed.
    event_names_set = set(raw.annotations.description)
    rename_events_set = set(cfg.rename_events.keys())
    events_not_in_raw = rename_events_set - event_names_set
    if events_not_in_raw:
        msg = (
            f"You requested to rename the following events, but "
            f"they are not present in the BIDS input data:\n"
            f"{', '.join(sorted(list(events_not_in_raw)))}"
        )
        if cfg.on_rename_missing_events == "warn":
            logger.warning(**gen_log_kwargs(message=msg))
        elif cfg.on_rename_missing_events == "raise":
            raise ValueError(msg)
        else:
            # should be guaranteed
            assert cfg.on_rename_missing_events == "ignore"

    # Do the actual event renaming.
    msg = "Renaming events …"
    logger.info(**gen_log_kwargs(message=msg))
    descriptions_list = list(raw.annotations.description)
    for old_event_name, new_event_name in cfg.rename_events.items():
        msg = f"… {old_event_name} -> {new_event_name}"
        logger.info(**gen_log_kwargs(message=msg))
        for idx, description in enumerate(descriptions_list):
            if description == old_event_name:
                descriptions_list[idx] = new_event_name

    raw.annotations.description = np.array(descriptions_list, dtype=str)


def _load_data(cfg: SimpleNamespace, bids_path: BIDSPath) -> mne.io.BaseRaw:
    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv

    subject = bids_path.subject
    raw = read_raw_bids(
        bids_path=bids_path,
        extra_params=cfg.reader_extra_params or {},
        verbose=cfg.read_raw_bids_verbose,
    )

    _crop_data(cfg, raw=raw, subject=subject)

    raw.load_data()
    if hasattr(raw, "fix_mag_coil_types"):
        raw.fix_mag_coil_types()

    return raw


def _crop_data(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
) -> None:
    """Crop the data to the desired duration.

    Modifies ``raw`` in-place.
    """
    if subject != "emptyroom" and cfg.crop_runs is not None:
        raw.crop(*cfg.crop_runs)


def _drop_channels_func(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
) -> None:
    """Drop channels from the data.

    Modifies ``raw`` in-place.
    """
    if cfg.drop_channels:
        msg = f"Dropping channels: {', '.join(cfg.drop_channels)}"
        logger.info(**gen_log_kwargs(message=msg))
        raw.drop_channels(cfg.drop_channels, on_missing="warn")


def _create_bipolar_channels(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str | None,
) -> None:
    """Create a channel from a bipolar referencing scheme..

    Modifies ``raw`` in-place.
    """
    if cfg.ch_types == ["eeg"] and cfg.eeg_bipolar_channels:
        msg = "Creating bipolar channels …"
        logger.info(**gen_log_kwargs(message=msg))
        for ch_name, (anode, cathode) in cfg.eeg_bipolar_channels.items():
            msg = f"    {anode} – {cathode} -> {ch_name}"
            logger.info(**gen_log_kwargs(message=msg))
            mne.set_bipolar_reference(
                raw,
                anode=anode,
                cathode=cathode,
                ch_name=ch_name,
                drop_refs=False,
                copy=False,
            )
        # If we created a new bipolar channel that the user wishes to
        # # use as an EOG channel, it is probably a good idea to set its
        # channel type to 'eog'. Bipolar channels, by default, don't have a
        # location, so one might get unexpected results otherwise, as the
        # channel would influence e.g. in GFP calculations, but not appear on
        # topographic maps.
        eog_chs_subj_sess = get_eog_channels(cfg.eog_channels, subject, session)
        if eog_chs_subj_sess and any(
            [
                eog_ch_name in cfg.eeg_bipolar_channels
                for eog_ch_name in eog_chs_subj_sess
            ]
        ):
            msg = "Setting channel type of new bipolar EOG channel(s) …"
            logger.info(**gen_log_kwargs(message=msg))
        if eog_chs_subj_sess is not None:
            for eog_ch_name in eog_chs_subj_sess:
                if eog_ch_name in cfg.eeg_bipolar_channels:
                    msg = f"    {eog_ch_name} -> eog"
                    logger.info(**gen_log_kwargs(message=msg))
                    raw.set_channel_types({eog_ch_name: "eog"})


def _set_eeg_montage(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str | None,
) -> None:
    """Set an EEG template montage if requested.

    Modifies ``raw`` in-place.
    """
    montage = cfg.eeg_template_montage
    if cfg.datatype == "eeg" and montage:
        msg = f"Setting EEG channel locations to template montage: {montage}."
        logger.info(**gen_log_kwargs(message=msg))
        raw.set_montage(montage, match_case=False, match_alias=True)


def _fix_stim_artifact_func(cfg: SimpleNamespace, raw: mne.io.BaseRaw) -> None:
    """Fix stimulation artifact in the data."""
    if not cfg.fix_stim_artifact:
        return

    events, _ = mne.events_from_annotations(raw)
    mne.preprocessing.fix_stim_artifact(
        raw,
        events=events,
        event_id=None,
        tmin=cfg.stim_artifact_tmin,
        tmax=cfg.stim_artifact_tmax,
        mode="linear",
    )


def import_experimental_data(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
    bids_path_bads_in: BIDSPath | None,
    data_is_rest: bool | None,
) -> mne.io.BaseRaw:
    """Run the data import.

    Parameters
    ----------
    cfg
        The local configuration.
    bids_path_in
        The BIDS path to the data to import.
    bids_path_bads_in
        The BIDS path to the bad channels file.
    data_is_rest : bool | None
        Whether the data is resting state data. If ``None``, ``cfg.task``
        is checked.

    Returns
    -------
    raw
        The imported data.
    """
    subject = bids_path_in.subject
    session = bids_path_in.session
    run = bids_path_in.run

    # 1. _load_data (_crop_data)
    raw = _load_data(cfg=cfg, bids_path=bids_path_in)
    # 2. _set_eeg_montage
    _set_eeg_montage(cfg=cfg, raw=raw, subject=subject, session=session, run=run)
    # 3. _create_bipolar_channels
    _create_bipolar_channels(
        cfg=cfg, raw=raw, subject=subject, session=session, run=run
    )
    # 4. _drop_channels_func
    _drop_channels_func(cfg=cfg, raw=raw, subject=subject, session=session)
    # 5. _find_breaks_func
    _find_breaks_func(cfg=cfg, raw=raw, subject=subject, session=session, run=run)
    if data_is_rest is None:
        data_is_rest = (cfg.task == "rest") or cfg.task_is_rest
    if not data_is_rest:
        # 6. _rename_events_func
        _rename_events_func(cfg=cfg, raw=raw, subject=subject, session=session, run=run)
        # 7. _fix_stim_artifact_func
        _fix_stim_artifact_func(cfg=cfg, raw=raw)

    if bids_path_bads_in is not None:
        run = "rest" if data_is_rest else run  # improve logging
        bads = _read_bads_tsv(cfg=cfg, bids_path_bads=bids_path_bads_in)
        msg = f"Marking {len(bads)} channel{_pl(bads)} as bad."
        logger.info(**gen_log_kwargs(message=msg))
        raw.info["bads"] = bads
        raw.info._check_consistency()

    return raw


def import_er_data(
    *,
    cfg: SimpleNamespace,
    bids_path_er_in: BIDSPath,
    bids_path_ref_in: BIDSPath | None,
    bids_path_er_bads_in: BIDSPath | None,
    bids_path_ref_bads_in: BIDSPath | None,
    prepare_maxwell_filter: bool,
) -> mne.io.BaseRaw:
    """Import empty-room data.

    Parameters
    ----------
    cfg
        The local configuration.
    bids_path_er_in
        The BIDS path to the empty room data.
    bids_path_ref_in
        The BIDS path to the reference data.
    bids_path_er_bads_in
        The BIDS path to the empty room bad channels file.
    bids_path_ref_bads_in
        The BIDS path to the reference data bad channels file.
    prepare_maxwell_filter
        Whether to prepare the empty-room data for Maxwell filtering.

    Returns
    -------
    raw_er
        The imported data.
    """
    raw_er = _load_data(cfg, bids_path_er_in)
    session = bids_path_er_in.session

    _drop_channels_func(cfg, raw=raw_er, subject="emptyroom", session=session)
    if bids_path_er_bads_in is not None:
        all_bads = _read_bads_tsv(
            cfg=cfg,
            bids_path_bads=bids_path_er_bads_in,
        )
        # There could be EEG channels in this list, so pick subset by name
        raw_er.info["bads"] = [bad for bad in all_bads if bad in raw_er.ch_names]

    # Don't deal with ref for now (initial data quality / auto bad step)
    if bids_path_ref_in is None:
        return raw_er

    # Load reference run plus its auto-bads
    raw_ref = read_raw_bids(
        bids_path_ref_in,
        extra_params=cfg.reader_extra_params or {},
        verbose=cfg.read_raw_bids_verbose,
    )
    if bids_path_ref_bads_in is not None:
        bads = _read_bads_tsv(
            cfg=cfg,
            bids_path_bads=bids_path_ref_bads_in,
        )
        raw_ref.info["bads"] = bads
        raw_ref.info._check_consistency()
    raw_ref.pick("meg")
    raw_er.pick("meg")

    if prepare_maxwell_filter:
        # We need to include any automatically found bad channels, if relevant.
        # TODO: This 'union' operation should affect the raw runs, too,
        # otherwise rank mismatches will still occur (eventually for some
        # configs). But at least using the union here should reduce them.
        raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(
            raw_er=raw_er,
            raw=raw_ref,
            bads="union",
        )
    else:
        # Take bads from the reference run
        raw_er.info["bads"] = raw_ref.info["bads"]

    return raw_er


def _find_breaks_func(
    *,
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str | None,
) -> None:
    if not cfg.find_breaks:
        return

    msg = f"Finding breaks with a minimum duration of {cfg.min_break_duration} seconds."
    logger.info(**gen_log_kwargs(message=msg))

    break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        min_break_duration=cfg.min_break_duration,
        t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
        t_stop_before_next=cfg.t_break_annot_stop_before_next_event,
    )

    msg = (
        f"Found and annotated "
        f"{len(break_annots) if break_annots else 'no'} break periods."
    )
    logger.info(**gen_log_kwargs(message=msg))

    raw.set_annotations(raw.annotations + break_annots)  # add to existing


def _get_bids_path_in(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    kind: RunKindT = "orig",
) -> BIDSPath:
    # b/c can be used before this is updated
    path_kwargs = dict(
        subject=subject,
        run=run,
        session=session,
        task=task or cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=get_datatype(config=cfg),
        check=False,
    )
    if kind != "orig":
        assert kind in ("sss", "filt"), kind
        path_kwargs["root"] = cfg.deriv_root
        path_kwargs["suffix"] = "raw"
        path_kwargs["extension"] = ".fif"
        path_kwargs["processing"] = kind
    else:
        path_kwargs["root"] = cfg.bids_root
        path_kwargs["suffix"] = None
        path_kwargs["extension"] = None
        path_kwargs["processing"] = cfg.proc
    bids_path_in = BIDSPath(**path_kwargs)
    return bids_path_in


def _get_run_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    kind: RunKindT,
    add_bads: bool = False,
    allow_missing: bool = False,
    key: str | None = None,
) -> InFilesT:
    bids_path_in = _get_bids_path_in(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind=kind,
    )
    return _path_dict(
        cfg=cfg,
        bids_path_in=bids_path_in,
        key=key,
        add_bads=add_bads,
        kind=kind,
        allow_missing=allow_missing,
        subject=subject,
        session=session,
    )


def _get_rest_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    kind: RunKindT,
    add_bads: bool = False,
) -> InFilesT:
    if not (cfg.process_rest and not cfg.task_is_rest):
        return dict()
    return _get_run_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=None,
        task="rest",
        kind=kind,
        add_bads=add_bads,
        allow_missing=True,
    )


def _get_noise_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    kind: RunKindT,
    mf_reference_run: str | None,
    add_bads: bool = False,
) -> InFilesT:
    if not (cfg.process_empty_room and get_datatype(config=cfg) == "meg"):
        return dict()
    if kind != "orig":
        assert kind in ("sss", "filt")
        raw_fname = _get_bids_path_in(
            cfg=cfg,
            subject=subject,
            session=session,
            run=None,
            task="noise",
            kind=kind,
        )
    else:
        # This must match the logic of _02_find_empty_room.py
        raw_fname = _get_bids_path_in(
            cfg=cfg,
            subject=subject,
            session=session,
            run=mf_reference_run,
            task=get_task(config=cfg),
            kind=kind,
        )
        raw_fname = _read_json(_empty_room_match_path(raw_fname, cfg))["fname"]
        if raw_fname is None:
            return dict()
        raw_fname = get_bids_path_from_fname(raw_fname)
    return _path_dict(
        cfg=cfg,
        bids_path_in=raw_fname,
        add_bads=add_bads,
        kind=kind,
        allow_missing=True,
        subject=subject,
        session=session,
    )


def _get_run_rest_noise_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    kind: RunKindT,
    mf_reference_run: str | None,
    add_bads: bool = False,
) -> InFilesT:
    if run is None and task in ("noise", "rest"):
        if task == "noise":
            path = _get_noise_path(
                mf_reference_run=mf_reference_run,
                cfg=cfg,
                subject=subject,
                session=session,
                kind=kind,
                add_bads=add_bads,
            )
        else:
            assert task == "rest"
            path = _get_rest_path(
                cfg=cfg,
                subject=subject,
                session=session,
                kind=kind,
                add_bads=add_bads,
            )
    else:
        path = _get_run_path(
            run=run,
            task=task,
            cfg=cfg,
            subject=subject,
            session=session,
            kind=kind,
            add_bads=add_bads,
        )
    return path


def _get_mf_reference_run_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    add_bads: bool = False,
) -> InFilesT:
    return _get_run_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=cfg.mf_reference_run,
        task=None,
        kind="orig",
        add_bads=add_bads,
        key="raw_ref_run",
    )


def _empty_room_match_path(run_path: BIDSPath, cfg: SimpleNamespace) -> BIDSPath:
    return run_path.copy().update(
        extension=".json", suffix="emptyroommatch", root=cfg.deriv_root
    )


def _path_dict(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
    add_bads: bool = False,
    kind: RunKindT,
    allow_missing: bool,
    key: str | None = None,
    subject: str,
    session: str | None,
) -> InFilesT:
    in_files = dict()
    key = key or f"raw_task-{bids_path_in.task}_run-{bids_path_in.run}"
    in_files[key] = bids_path_in
    _update_for_splits(in_files, key, single=True, allow_missing=True)
    if allow_missing and not in_files[key].fpath.exists():
        return dict()
    if add_bads:
        bads_tsv_fname = _bads_path(
            cfg=cfg,
            bids_path_in=bids_path_in,
            subject=subject,
            session=session,
        )
        if bads_tsv_fname.fpath.is_file() or not allow_missing:
            in_files[f"{key}-bads"] = bads_tsv_fname
    return in_files


def _bads_path(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
    subject: str,
    session: str | None,
) -> BIDSPath:
    return bids_path_in.copy().update(
        suffix="bads",
        extension=".tsv",
        root=cfg.deriv_root,
        subject=subject,
        session=session,
        split=None,
        check=False,
    )


def _read_bads_tsv(
    *,
    cfg: SimpleNamespace,
    bids_path_bads: BIDSPath,
) -> list[str]:
    bads_tsv = pd.read_csv(bids_path_bads.fpath, sep="\t", header=0)
    out = bads_tsv[bads_tsv.columns[0]].tolist()
    assert isinstance(out, list)
    assert all(isinstance(o, str) for o in out)
    return out


def _import_data_kwargs(*, config: SimpleNamespace, subject: str) -> dict[str, Any]:
    """Get config params needed for any raw data loading."""
    return dict(
        # import_experimental_data / general
        process_empty_room=config.process_empty_room,
        process_rest=config.process_rest,
        task_is_rest=config.task_is_rest,
        # _get_raw_paths, _get_noise_path
        use_maxwell_filter=config.use_maxwell_filter,
        mf_reference_run=get_mf_reference_run(config=config),
        data_type=config.data_type,
        # automatic add_bads
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        find_flat_channels_meg=config.find_flat_channels_meg,
        find_bad_channels_extra_kws=config.find_bad_channels_extra_kws,
        # 1. _load_data
        reader_extra_params=config.reader_extra_params,
        crop_runs=config.crop_runs,
        read_raw_bids_verbose=config.read_raw_bids_verbose,
        # 2. _set_eeg_montage
        eeg_template_montage=config.eeg_template_montage,
        # 3. _create_bipolar_channels
        eeg_bipolar_channels=config.eeg_bipolar_channels,
        ch_types=config.ch_types,
        eog_channels=config.eog_channels,
        # 4. _drop_channels_func
        drop_channels=config.drop_channels,
        # 5. _find_breaks_func
        find_breaks=config.find_breaks,
        min_break_duration=config.min_break_duration,
        t_break_annot_start_after_previous_event=config.t_break_annot_start_after_previous_event,  # noqa:E501
        t_break_annot_stop_before_next_event=config.t_break_annot_stop_before_next_event,  # noqa:E501
        # 6. _rename_events_func
        rename_events=config.rename_events,
        on_rename_missing_events=config.on_rename_missing_events,
        # 7. _fix_stim_artifact_func
        fix_stim_artifact=config.fix_stim_artifact,
        stim_artifact_tmin=config.stim_artifact_tmin,
        stim_artifact_tmax=config.stim_artifact_tmax,
        # args used for all runs that process raw (reporting / writing)
        plot_psd_for_runs=config.plot_psd_for_runs,
        _raw_split_size=config._raw_split_size,
        runs=get_runs(config=config, subject=subject),  # XXX needs to accept session!
        **_bids_kwargs(config=config),
    )


def _read_raw_msg(
    bids_path_in: BIDSPath,
    run: str | None,
    task: str | None,
) -> tuple[str, RunTypeT]:
    run_type: RunTypeT = "experimental"
    if run is None and task in ("noise", "rest"):
        if task == "noise":
            run_type = "empty-room"
        else:
            run_type = "resting-state"
    return f"Reading {run_type} recording: {bids_path_in.basename}", run_type
