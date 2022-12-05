from types import SimpleNamespace
from typing import Dict, Optional, Iterable, Union, List, Literal

import mne
from mne_bids import BIDSPath, read_raw_bids, get_bids_path_from_fname
import numpy as np
import pandas as pd

from ._io import _read_json, _empty_room_match_path
from ._logging import gen_log_kwargs, logger
from ._run import _update_for_splits
from .typing import PathLike


def make_epochs(
    *,
    task: str,
    subject: str,
    session: Optional[str],
    raw: mne.io.BaseRaw,
    event_id: Optional[Union[Dict[str, int], Literal['auto']]],
    conditions: Union[Iterable[str], Dict[str, str]],
    tmin: float,
    tmax: float,
    metadata_tmin: Optional[float],
    metadata_tmax: Optional[float],
    metadata_keep_first: Optional[Iterable[str]],
    metadata_keep_last: Optional[Iterable[str]],
    metadata_query: Optional[str],
    event_repeated: Literal['error', 'drop', 'merge'],
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
        assert tmin == 0., "epochs_tmin must be 0 for rest"
        assert rest_epochs_overlap is not None, \
            "epochs_overlap cannot be None for rest"
        events = mne.make_fixed_length_events(
            raw, id=3000, start=0,
            duration=rest_epochs_duration,
            overlap=rest_epochs_overlap,
            stop=stop)
        event_id = dict(rest=3000)
        metadata = None
    else:  # Events for task runs
        if event_id is None:
            event_id = 'auto'

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
            event_names=event_id,
            keys=conditions
        )

        if metadata_tmin is None:
            metadata_tmin = tmin
        if metadata_tmax is None:
            metadata_tmax = tmax

        # The returned `events` and `event_id` will only contain
        # the events from `row_event_names` – which is basically equivalent to
        # what the user requested via `config.conditions` (only with potential
        # nested event names expanded, e.g. `visual` might now be
        # `visual/left` and `visual/right`)
        metadata, events, event_id = mne.epochs.make_metadata(
            row_events=row_event_names,
            events=events, event_id=event_id,
            tmin=metadata_tmin, tmax=metadata_tmax,
            keep_first=metadata_keep_first,
            keep_last=metadata_keep_last,
            sfreq=raw.info['sfreq']
        )

    # Epoch the data
    # Do not reject based on peak-to-peak or flatness thresholds at this stage
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        proj=False, baseline=None,
                        preload=False, decim=epochs_decim,
                        metadata=metadata,
                        event_repeated=event_repeated,
                        reject=None)

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
            idx_keep = epochs.metadata.eval(metadata_query, engine='python')
        except pandas.core.computation.ops.UndefinedVariableError:
            msg = (f'Metadata query failed to select any columns: '
                   f'{metadata_query}')
            logger.warn(**gen_log_kwargs(message=msg))
            return epochs

        idx_drop = epochs.metadata.index[~idx_keep]
        epochs.drop(
            indices=idx_drop,
            reason='metadata query',
            verbose=False
        )
        del idx_keep, idx_drop

    return epochs


def annotations_to_events(
    *,
    raw_paths: List[PathLike]
) -> Dict[str, int]:
    """Generate a unique event name -> event code mapping.

    The mapping can that can be used across all passed raws.
    """
    event_names: List[str] = []
    for raw_fname in raw_paths:
        raw = mne.io.read_raw_fif(raw_fname)
        _, event_id = mne.events_from_annotations(raw=raw)
        for event_name in event_id.keys():
            if event_name not in event_names:
                event_names.append(event_name)

    event_names = sorted(event_names)
    event_name_to_code_map = {
        name: code
        for code, name in enumerate(event_names, start=1)
    }

    return event_name_to_code_map


def _rename_events_func(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
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
        msg = (f'You requested to rename the following events, but '
               f'they are not present in the BIDS input data:\n'
               f'{", ".join(sorted(list(events_not_in_raw)))}')
        if cfg.on_rename_missing_events == 'warn':
            logger.warning(**gen_log_kwargs(message=msg))
        elif cfg.on_rename_missing_events == 'raise':
            raise ValueError(msg)
        else:
            # should be guaranteed
            assert cfg.on_rename_missing_events == 'ignore'

    # Do the actual event renaming.
    msg = 'Renaming events …'
    logger.info(**gen_log_kwargs(message=msg))
    descriptions = list(raw.annotations.description)
    for old_event_name, new_event_name in cfg.rename_events.items():
        msg = f'… {old_event_name} -> {new_event_name}'
        logger.info(**gen_log_kwargs(message=msg))
        for idx, description in enumerate(descriptions.copy()):
            if description == old_event_name:
                descriptions[idx] = new_event_name

    descriptions = np.asarray(descriptions, dtype=str)
    raw.annotations.description = descriptions


def _load_data(cfg: SimpleNamespace, bids_path: BIDSPath) -> mne.io.BaseRaw:
    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv

    subject = bids_path.subject
    raw = read_raw_bids(bids_path=bids_path,
                        extra_params=cfg.reader_extra_params)

    _crop_data(cfg, raw=raw, subject=subject)

    raw.load_data()
    if hasattr(raw, 'fix_mag_coil_types'):
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
    if subject != 'emptyroom' and cfg.crop_runs is not None:
        raw.crop(*cfg.crop_runs)


def _drop_channels_func(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
) -> None:
    """Drop channels from the data.

    Modifies ``raw`` in-place.
    """
    if cfg.drop_channels:
        msg = f'Dropping channels: {", ".join(cfg.drop_channels)}'
        logger.info(**gen_log_kwargs(message=msg))
        raw.drop_channels(cfg.drop_channels, on_missing='warn')


def _create_bipolar_channels(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str]
) -> None:
    """Create a channel from a bipolar referencing scheme..

    Modifies ``raw`` in-place.
    """
    if cfg.ch_types == ['eeg'] and cfg.eeg_bipolar_channels:
        msg = 'Creating bipolar channels …'
        logger.info(**gen_log_kwargs(message=msg))
        for ch_name, (anode, cathode) in cfg.eeg_bipolar_channels.items():
            msg = f'    {anode} – {cathode} -> {ch_name}'
            logger.info(**gen_log_kwargs(message=msg))
            mne.set_bipolar_reference(raw, anode=anode, cathode=cathode,
                                      ch_name=ch_name, drop_refs=False,
                                      copy=False)
        # If we created a new bipolar channel that the user wishes to
        # # use as an EOG channel, it is probably a good idea to set its
        # channel type to 'eog'. Bipolar channels, by default, don't have a
        # location, so one might get unexpected results otherwise, as the
        # channel would influence e.g. in GFP calculations, but not appear on
        # topographic maps.
        if (cfg.eog_channels and
                any([eog_ch_name in cfg.eeg_bipolar_channels
                     for eog_ch_name in cfg.eog_channels])):
            msg = 'Setting channel type of new bipolar EOG channel(s) …'
            logger.info(**gen_log_kwargs(message=msg))
        for eog_ch_name in cfg.eog_channels:
            if eog_ch_name in cfg.eeg_bipolar_channels:
                msg = f'    {eog_ch_name} -> eog'
                logger.info(**gen_log_kwargs(message=msg))
                raw.set_channel_types({eog_ch_name: 'eog'})


def _set_eeg_montage(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str]
) -> None:
    """Set an EEG template montage if requested.

    Modifies ``raw`` in-place.
    """
    montage = cfg.eeg_template_montage
    is_mne_montage = isinstance(montage,
                                mne.channels.montage.DigMontage)
    montage_name = 'custom_montage' if is_mne_montage else montage
    if cfg.datatype == 'eeg' and montage:
        msg = (f'Setting EEG channel locations to template montage: '
               f'{montage}.')
        logger.info(**gen_log_kwargs(message=msg))
        if not is_mne_montage:
            montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False, on_missing='warn')


def _fix_stim_artifact_func(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw
) -> None:
    """Fix stimulation artifact in the data."""
    if not cfg.fix_stim_artifact:
        return

    events, _ = mne.events_from_annotations(raw)
    mne.preprocessing.fix_stim_artifact(
        raw, events=events, event_id=None,
        tmin=cfg.stim_artifact_tmin,
        tmax=cfg.stim_artifact_tmax,
        mode='linear'
    )


def import_experimental_data(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
    bids_path_bads_in: Optional[BIDSPath],
    data_is_rest: Optional[bool],
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

    raw = _load_data(cfg=cfg, bids_path=bids_path_in)
    _set_eeg_montage(
        cfg=cfg, raw=raw, subject=subject, session=session, run=run
    )
    _create_bipolar_channels(cfg=cfg, raw=raw, subject=subject,
                             session=session, run=run)
    _drop_channels_func(cfg=cfg, raw=raw, subject=subject, session=session)
    _find_breaks_func(cfg=cfg, raw=raw, subject=subject, session=session,
                      run=run)
    if data_is_rest is None:
        data_is_rest = (cfg.task == 'rest') or cfg.task_is_rest
    if not data_is_rest:
        _rename_events_func(
            cfg=cfg, raw=raw, subject=subject, session=session, run=run
        )
        _fix_stim_artifact_func(cfg=cfg, raw=raw)

    if bids_path_bads_in is not None:
        bads = _read_bads_tsv(cfg=cfg, bids_path_bads=bids_path_bads_in)
        msg = f'Marking {len(bads)} channels as bad.'
        logger.info(**gen_log_kwargs(message=msg))
        raw.info['bads'] = bads
        raw.info._check_consistency()

    return raw


def import_er_data(
    *,
    cfg: SimpleNamespace,
    bids_path_er_in: BIDSPath,
    bids_path_ref_in: Optional[BIDSPath],
    bids_path_er_bads_in: Optional[BIDSPath],
    bids_path_ref_bads_in: Optional[BIDSPath],
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

    Returns
    -------
    raw_er
        The imported data.
    """
    raw_er = _load_data(cfg, bids_path_er_in)
    session = bids_path_er_in.session

    _drop_channels_func(cfg, raw=raw_er, subject='emptyroom', session=session)
    if bids_path_er_bads_in is not None:
        raw_er.info['bads'] = _read_bads_tsv(
            cfg=cfg,
            bids_path_bads=bids_path_er_bads_in,
        )
    raw_er.pick_types(meg=True, exclude=[])

    # Don't deal with ref for now (initial data quality / auto bad step)
    if bids_path_ref_in is None:
        return raw_er

    # Load reference run plus its auto-bads
    raw_ref = read_raw_bids(bids_path_ref_in,
                            extra_params=cfg.reader_extra_params)
    if bids_path_ref_bads_in is not None:
        bads = _read_bads_tsv(
            cfg=cfg,
            bids_path_bads=bids_path_ref_bads_in,
        )
        raw_ref.info['bads'] = bads
        raw_ref.info._check_consistency()
    raw_ref.pick_types(meg=True, exclude=[])

    if prepare_maxwell_filter:
        # We need to include any automatically found bad channels, if relevant.
        # TODO: This 'union' operation should affect the raw runs, too,
        # otherwise rank mismatches will still occur (eventually for some
        # configs). But at least using the union here should reduce them.
        raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(
            raw_er=raw_er,
            raw=raw_ref,
            bads='union',
        )
    else:
        # Take bads from the reference run
        raw_er.info['bads'] = raw_ref.info['bads']

    return raw_er


def _find_breaks_func(
    *,
    cfg,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
) -> None:
    if not cfg.find_breaks:
        return

    msg = (f'Finding breaks with a minimum duration of '
           f'{cfg.min_break_duration} seconds.')
    logger.info(**gen_log_kwargs(message=msg))

    break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        min_break_duration=cfg.min_break_duration,
        t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
        t_stop_before_next=cfg.t_break_annot_stop_before_next_event
    )

    msg = (f'Found and annotated '
           f'{len(break_annots) if break_annots else "no"} break periods.')
    logger.info(**gen_log_kwargs(message=msg))

    raw.set_annotations(raw.annotations + break_annots)  # add to existing


def _get_raw_paths(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    kind: Literal['raw', 'sss'],
    add_bads: bool = True,
    include_mf_ref: bool = True,
) -> dict:
    # Construct the basenames of the files we wish to load, and of the empty-
    # room recording we wish to save.
    # The basenames of the empty-room recording output file does not contain
    # the "run" entity.
    path_kwargs = dict(
        subject=subject,
        run=run,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        check=False
    )
    if kind == 'sss':
        path_kwargs['root'] = cfg.deriv_root
        path_kwargs['suffix'] = 'raw'
        path_kwargs['extension'] = '.fif'
        path_kwargs['processing'] = 'sss'
    else:
        assert kind == 'orig', kind
        path_kwargs['root'] = cfg.bids_root
        path_kwargs['suffix'] = None
        path_kwargs['extension'] = None
        path_kwargs['processing'] = cfg.proc
    bids_path_in = BIDSPath(**path_kwargs)

    in_files = dict()
    key = f'raw_run-{run}'
    in_files[key] = bids_path_in
    _update_for_splits(in_files, key, single=True)
    if add_bads:
        _add_bads_file(
            cfg=cfg,
            in_files=in_files,
            key=key,
        )
    orig_key = key

    if run == cfg.runs[0]:
        do = dict(
            rest=cfg.process_rest and not cfg.task_is_rest,
            noise=cfg.process_empty_room and cfg.datatype == 'meg',
        )
        for task in ('rest', 'noise'):
            if not do[task]:
                continue
            key = f'raw_{task}'
            if kind == 'sss':
                raw_fname = bids_path_in.copy().update(
                    run=None, task=task)
            else:
                if task == 'rest':
                    raw_fname = bids_path_in.copy().update(
                        run=None, task=task)
                else:
                    raw_fname = _read_json(
                        _empty_room_match_path(bids_path_in, cfg))['fname']
                    if raw_fname is not None:
                        raw_fname = get_bids_path_from_fname(raw_fname)
            if raw_fname is None:
                continue
            in_files[key] = raw_fname
            _update_for_splits(
                in_files, key, single=True, allow_missing=True)
            if not in_files[key].fpath.exists():
                in_files.pop(key)
            elif add_bads:
                _add_bads_file(
                    cfg=cfg,
                    in_files=in_files,
                    key=key,
                )
            if include_mf_ref and task == 'noise':
                key = 'raw_ref_run'
                in_files[key] = in_files[orig_key].copy().update(
                    run=cfg.mf_reference_run)
                _update_for_splits(
                    in_files, key, single=True, allow_missing=True)
                if not in_files[key].fpath.exists():
                    in_files.pop(key)
                elif add_bads:
                    _add_bads_file(
                        cfg=cfg,
                        in_files=in_files,
                        key=key,
                    )

    return in_files


def _add_bads_file(
    *,
    cfg: SimpleNamespace,
    in_files: dict,
    key: str,
) -> None:
    bids_path_in = in_files[key]
    bads_tsv_fname = _bads_path(
        cfg=cfg,
        bids_path_in=bids_path_in,
        )
    if bads_tsv_fname.fpath.is_file():
        in_files[f'{key}-bads'] = bads_tsv_fname


def _auto_scores_path(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
) -> BIDSPath:
    return bids_path_in.copy().update(
        suffix='scores',
        extension='.json',
        root=cfg.deriv_root,
        split=None,
        check=False,
    )


def _bads_path(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
) -> BIDSPath:
    return bids_path_in.copy().update(
        suffix='bads',
        extension='.tsv',
        root=cfg.deriv_root,
        split=None,
        check=False,
    )


def _read_bads_tsv(
    *,
    cfg: SimpleNamespace,
    bids_path_bads: BIDSPath,
) -> List[str]:
    bads_tsv = pd.read_csv(bids_path_bads.fpath, sep='\t', header=0)
    return bads_tsv[bads_tsv.columns[0]].tolist()
