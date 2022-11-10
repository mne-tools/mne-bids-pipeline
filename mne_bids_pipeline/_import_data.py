import copy
from types import SimpleNamespace
from typing import Dict, Optional, Iterable, Union, List

import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd

from ._config_utils import get_channels_to_analyze, get_task
from ._io import _write_json
from ._logging import gen_log_kwargs, logger
from ._typing import PathLike, Literal
from ._viz import plot_auto_scores


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
    decim: int,
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
                        preload=False, decim=decim,
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
            logger.warn(**gen_log_kwargs(message=msg, subject=subject,
                                         session=session))
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
            logger.warning(
                **gen_log_kwargs(message=msg, subject=subject, session=session)
            )
        elif cfg.on_rename_missing_events == 'raise':
            raise ValueError(msg)
        else:
            # should be guaranteed
            assert cfg.on_rename_missing_events == 'ignore'

    # Do the actual event renaming.
    msg = 'Renaming events …'
    logger.info(**gen_log_kwargs(
        message=msg, subject=subject, session=session, run=run)
    )
    descriptions = list(raw.annotations.description)
    for old_event_name, new_event_name in cfg.rename_events.items():
        msg = f'… {old_event_name} -> {new_event_name}'
        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run)
        )
        for idx, description in enumerate(descriptions.copy()):
            if description == old_event_name:
                descriptions[idx] = new_event_name

    descriptions = np.asarray(descriptions, dtype=str)
    raw.annotations.description = descriptions


def _find_bad_channels(
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
) -> None:
    """Find and mark bad MEG channels.

    Modifies ``raw`` in-place.
    """
    if not (cfg.find_flat_channels_meg or cfg.find_noisy_channels_meg):
        return

    if (cfg.find_flat_channels_meg and
            not cfg.find_noisy_channels_meg):
        msg = 'Finding flat channels.'
    elif (cfg.find_noisy_channels_meg and
            not cfg.find_flat_channels_meg):
        msg = 'Finding noisy channels using Maxwell filtering.'
    else:
        msg = ('Finding flat channels, and noisy channels using '
               'Maxwell filtering.')

    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run))

    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=task,
                         run=run,
                         acquisition=cfg.acq,
                         processing=cfg.proc,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix=cfg.datatype,
                         datatype=cfg.datatype,
                         root=cfg.deriv_root)

    # Filter the data manually before passing it to find_bad_channels_maxwell()
    # This reduces memory usage, as we can control the number of jobs used
    # during filtering.
    raw_filt = raw.copy().filter(l_freq=None, h_freq=40, n_jobs=1)

    auto_noisy_chs, auto_flat_chs, auto_scores = \
        mne.preprocessing.find_bad_channels_maxwell(
            raw=raw_filt,
            calibration=cfg.mf_cal_fname,
            cross_talk=cfg.mf_ctc_fname,
            origin=cfg.mf_head_origin,
            coord_frame='head',
            return_scores=True,
            h_freq=None  # we filtered manually above
        )
    del raw_filt

    preexisting_bads = raw.info['bads'].copy()
    bads = preexisting_bads.copy()

    if cfg.find_flat_channels_meg:
        if auto_flat_chs:
            msg = (f'Found {len(auto_flat_chs)} flat channels: '
                   f'{", ".join(auto_flat_chs)}')
        else:
            msg = 'Found no flat channels.'
        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run)
        )
        bads.extend(auto_flat_chs)

    if cfg.find_noisy_channels_meg:
        if auto_noisy_chs:
            msg = (f'Found {len(auto_noisy_chs)} noisy channels: '
                   f'{", ".join(auto_noisy_chs)}')
        else:
            msg = 'Found no noisy channels.'

        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run)
        )
        bads.extend(auto_noisy_chs)

    bads = sorted(set(bads))
    raw.info['bads'] = bads
    msg = f'Marked {len(raw.info["bads"])} channels as bad.'
    logger.info(**gen_log_kwargs(
        message=msg, subject=subject, session=session, run=run)
    )

    if cfg.find_noisy_channels_meg:
        auto_scores_fname = bids_path.copy().update(
            suffix='scores', extension='.json', check=False)
        # TODO: This should be in our list of output files!
        _write_json(auto_scores_fname, auto_scores)

        if cfg.interactive:
            import matplotlib.pyplot as plt
            plot_auto_scores(auto_scores, ch_types=cfg.ch_types)
            plt.show()

    # Write the bad channels to disk.
    # TODO: This should also be in our list of output files
    bads_tsv_fname = bids_path.copy().update(suffix='bads',
                                             extension='.tsv',
                                             check=False)
    bads_for_tsv = []
    reasons = []

    if cfg.find_flat_channels_meg:
        bads_for_tsv.extend(auto_flat_chs)
        reasons.extend(['auto-flat'] * len(auto_flat_chs))
        preexisting_bads = set(preexisting_bads) - set(auto_flat_chs)

    if cfg.find_noisy_channels_meg:
        bads_for_tsv.extend(auto_noisy_chs)
        reasons.extend(['auto-noisy'] * len(auto_noisy_chs))
        preexisting_bads = set(preexisting_bads) - set(auto_noisy_chs)

    preexisting_bads = list(preexisting_bads)
    if preexisting_bads:
        bads_for_tsv.extend(preexisting_bads)
        reasons.extend(['pre-existing (before MNE-BIDS-pipeline was run)'] *
                       len(preexisting_bads))

    tsv_data = pd.DataFrame(dict(name=bads_for_tsv, reason=reasons))
    tsv_data = tsv_data.sort_values(by='name')
    tsv_data.to_csv(bads_tsv_fname, sep='\t', index=False)


def _load_data(cfg: SimpleNamespace, bids_path: BIDSPath) -> mne.io.BaseRaw:
    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv

    subject = bids_path.subject
    raw = read_raw_bids(bids_path=bids_path,
                        extra_params=cfg.reader_extra_params)

    # Save only the channel types we wish to analyze (including the
    # channels marked as "bad").
    if not cfg.use_maxwell_filter:
        picks = get_channels_to_analyze(raw.info, cfg)
        raw.pick(picks)

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
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))
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
        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run)
        )
        for ch_name, (anode, cathode) in cfg.eeg_bipolar_channels.items():
            msg = f'    {anode} – {cathode} -> {ch_name}'
            logger.info(**gen_log_kwargs(
                message=msg, subject=subject, session=session, run=run)
            )
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
            logger.info(**gen_log_kwargs(
                message=msg, subject=subject, session=session, run=run)
            )
        for eog_ch_name in cfg.eog_channels:
            if eog_ch_name in cfg.eeg_bipolar_channels:
                msg = f'    {eog_ch_name} -> eog'
                logger.info(**gen_log_kwargs(
                    message=msg, subject=subject, session=session, run=run)
                )
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
        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run)
        )
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
) -> mne.io.BaseRaw:
    """Run the data import.

    Parameters
    ----------
    cfg
        The local configuration.
    bids_path_in
        The BIDS path to the data to import.

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
    if cfg.task != "rest":
        _rename_events_func(
            cfg=cfg, raw=raw, subject=subject, session=session, run=run
        )
        _fix_stim_artifact_func(cfg=cfg, raw=raw)
    _find_bad_channels(cfg=cfg, raw=raw, subject=subject, session=session,
                       task=get_task(cfg), run=run)

    return raw


def import_er_data(
    *,
    cfg: SimpleNamespace,
    bids_path_er_in: BIDSPath,
    bids_path_ref_in: BIDSPath,
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

    Returns
    -------
    raw_er
        The imported data.
    """
    raw_er = _load_data(cfg, bids_path_er_in)
    session = bids_path_er_in.session

    _drop_channels_func(cfg, raw=raw_er, subject='emptyroom', session=session)

    # Only keep MEG channels.
    raw_er.pick_types(meg=True, exclude=[])

    # TODO: This 'union' operation should affect the raw runs, too, otherwise
    # rank mismatches will still occur (eventually for some configs).
    # But at least using the union here should reduce them.
    # TODO: We should also uso automatic bad finding on the empty room data
    raw_ref = read_raw_bids(bids_path_ref_in,
                            extra_params=cfg.reader_extra_params)
    if cfg.use_maxwell_filter:
        # We need to include any automatically found bad channels, if relevant.
        # TODO this is a bit of a hack because we don't use "in_files" access
        # here, but this is *in the same step where this file is generated*
        # so we cannot / should not put it in `in_files`.
        if cfg.find_flat_channels_meg or cfg.find_noisy_channels_meg:
            # match filename from _find_bad_channels
            bads_tsv_fname = bids_path_ref_in.copy().update(
                suffix='bads', extension='.tsv', root=cfg.deriv_root,
                check=False)
            bads_tsv = pd.read_csv(bads_tsv_fname.fpath, sep='\t', header=0)
            bads_tsv = bads_tsv[bads_tsv.columns[0]].tolist()
            raw_ref.info['bads'] = sorted(
                set(raw_ref.info['bads']) | set(bads_tsv))
            raw_ref.info._check_consistency()
        raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(
            raw_er=raw_er,
            raw=raw_ref,
            bads='union',
        )
    else:
        # Set same set of bads as in the reference run, but only for MEG
        # channels (we might not have non-MEG channels in empty-room
        # recordings).
        raw_er.info['bads'] = [ch for ch in raw_ref.info['bads']
                               if ch.startswith('MEG')]

    return raw_er


def import_rest_data(
    *,
    cfg: SimpleNamespace,
    bids_path_in: BIDSPath,
) -> mne.io.BaseRaw:
    """Import resting-state data for use as a noise source.

    Parameters
    ----------
    cfg
        The local configuration.
    bids_path_in : BIDSPath
        The path.

    Returns
    -------
    raw_rest
        The imported data.
    """
    cfg = copy.deepcopy(cfg)
    cfg.task = 'rest'

    raw_rest = import_experimental_data(
        cfg=cfg, bids_path_in=bids_path_in,
    )
    return raw_rest


def _find_breaks_func(
    *,
    cfg,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str],
) -> None:
    if not cfg.find_breaks:
        msg = 'Finding breaks has been disabled by the user.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        return

    msg = (f'Finding breaks with a minimum duration of '
           f'{cfg.min_break_duration} seconds.')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run))

    break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        min_break_duration=cfg.min_break_duration,
        t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
        t_stop_before_next=cfg.t_break_annot_stop_before_next_event
    )

    msg = (f'Found and annotated '
           f'{len(break_annots) if break_annots else "no"} break periods.')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run))

    raw.set_annotations(raw.annotations + break_annots)  # add to existing
