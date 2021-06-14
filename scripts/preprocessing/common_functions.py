from typing import Optional, List
import logging

import numpy as np
import pandas as pd
import json_tricks

import mne
from mne.preprocessing import find_bad_channels_maxwell
from mne_bids import BIDSPath, read_raw_bids

import config
from config import gen_log_message, get_mf_cal_fname, get_mf_ctc_fname

logger = logging.getLogger('mne-bids-pipeline')


def rename_events(raw, subject, session) -> None:
    """Rename events (actually, annotations descriptions) in ``raw``.

    Modifies ``raw`` in-place.
    """
    if not config.rename_events:
        return

    # Check if the user requested to rename events that don't exist.
    # We don't want this to go unnoticed.
    event_names_set = set(raw.annotations.description)
    rename_events_set = set(config.rename_events.keys())
    events_not_in_raw = rename_events_set - event_names_set
    if events_not_in_raw:
        msg = (f'You requested to rename the following events, but '
               f'they are not present in the BIDS input data:\n'
               f'{", ".join(sorted(list(events_not_in_raw)))}')
        if config.on_rename_missing_events == 'warn':
            logger.warning(msg)
        else:
            raise ValueError(msg)

    # Do the actual event renaming.
    msg = 'Renaming events …'
    logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                session=session))
    descriptions = list(raw.annotations.description)
    for old_event_name, new_event_name in config.rename_events.items():
        msg = f'… {old_event_name} -> {new_event_name}'
        logger.info(gen_log_message(message=msg, step=0,
                                    subject=subject, session=session))
        for idx, description in enumerate(descriptions.copy()):
            if description == old_event_name:
                descriptions[idx] = new_event_name

    descriptions = np.asarray(descriptions, dtype=str)
    raw.annotations.description = descriptions


def find_bad_channels(raw, subject, session, task, run) -> None:
    """Find and mark bad MEG channels.

    Modifies ``raw`` in-place.
    """
    if not (config.find_flat_channels_meg or config.find_noisy_channels_meg):
        return

    if (config.find_flat_channels_meg and
            not config.find_noisy_channels_meg):
        msg = 'Finding flat channels.'
    elif (config.find_noisy_channels_meg and
            not config.find_flat_channels_meg):
        msg = 'Finding noisy channels using Maxwell filtering.'
    else:
        msg = ('Finding flat channels, and noisy channels using '
               'Maxwell filtering.')

    logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                session=session))

    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=task,
                         run=run,
                         acquisition=config.acq,
                         processing=config.proc,
                         recording=config.rec,
                         space=config.space,
                         suffix=config.get_datatype(),
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root())

    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw=raw,
        calibration=get_mf_cal_fname(subject, session),
        cross_talk=get_mf_ctc_fname(subject, session),
        origin=config.mf_head_origin,
        coord_frame='head',
        return_scores=True)

    preexisting_bads = raw.info['bads'].copy()
    bads = preexisting_bads.copy()

    if config.find_flat_channels_meg:
        msg = f'Found {len(auto_flat_chs)} flat channels.'
        logger.info(gen_log_message(message=msg, step=0,
                                    subject=subject, session=session))
        bads.extend(auto_flat_chs)
    if config.find_noisy_channels_meg:
        msg = f'Found {len(auto_noisy_chs)} noisy channels.'
        logger.info(gen_log_message(message=msg, step=0,
                                    subject=subject, session=session))
        bads.extend(auto_noisy_chs)

    bads = sorted(set(bads))
    raw.info['bads'] = bads
    msg = f'Marked {len(raw.info["bads"])} channels as bad.'
    logger.info(gen_log_message(message=msg, step=0,
                                subject=subject, session=session))

    if config.find_noisy_channels_meg:
        auto_scores_fname = bids_path.copy().update(
            suffix='scores', extension='.json', check=False)
        with open(auto_scores_fname, 'w') as f:
            json_tricks.dump(auto_scores, fp=f, allow_nan=True,
                             sort_keys=False)

        if config.interactive:
            import matplotlib.pyplot as plt
            config.plot_auto_scores(auto_scores)
            plt.show()

    # Write the bad channels to disk.
    bads_tsv_fname = bids_path.copy().update(suffix='bads',
                                             extension='.tsv',
                                             check=False)
    bads_for_tsv = []
    reasons = []

    if config.find_flat_channels_meg:
        bads_for_tsv.extend(auto_flat_chs)
        reasons.extend(['auto-flat'] * len(auto_flat_chs))
        preexisting_bads = set(preexisting_bads) - set(auto_flat_chs)

    if config.find_noisy_channels_meg:
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


def load_data(bids_path):
    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv

    subject = bids_path.subject
    raw = read_raw_bids(bids_path=bids_path)

    # Save only the channel types we wish to analyze (including the
    # channels marked as "bad").
    if not config.use_maxwell_filter:
        picks = config.get_channels_to_analyze(raw.info)
        raw.pick(picks)

    crop_data(raw=raw, subject=subject)

    raw.load_data()
    if hasattr(raw, 'fix_mag_coil_types'):
        raw.fix_mag_coil_types()

    return raw


def crop_data(raw, subject):
    """Crop the data to the desired duration.

    Modifies ``raw`` in-place.
    """
    if subject != 'emptyroom' and config.crop_runs is not None:
        raw.crop(*config.crop_runs)


def drop_channels(raw, subject, session) -> None:
    """Drop channels from the data.

    Modifies ``raw`` in-place.
    """
    if config.drop_channels:
        msg = f'Dropping channels: {", ".join(config.drop_channels)}'
        logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                    session=session))
        raw.drop_channels(config.drop_channels)


def create_bipolar_channels(raw, subject, session) -> None:
    """Create a channel from a bipolar referencing scheme..

    Modifies ``raw`` in-place.
    """
    if config.ch_types == ['eeg'] and config.eeg_bipolar_channels:
        msg = 'Creating bipolar channels …'
        logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                    session=session))
        raw.load_data()
        for ch_name, (anode, cathode) in config.eeg_bipolar_channels.items():
            msg = f'    {anode} – {cathode} -> {ch_name}'
            logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                        session=session))
            mne.set_bipolar_reference(raw, anode=anode, cathode=cathode,
                                      ch_name=ch_name, drop_refs=False,
                                      copy=False)
        # If we created a new bipolar channel that the user wishes to
        # # use as an EOG channel, it is probably a good idea to set its
        # channel type to 'eog'. Bipolar channels, by default, don't have a
        # location, so one might get unexpected results otherwise, as the
        # channel would influence e.g. in GFP calculations, but not appear on
        # topographic maps.
        if (config.eog_channels and
                any([eog_ch_name in config.eeg_bipolar_channels
                     for eog_ch_name in config.eog_channels])):
            msg = 'Setting channel type of new bipolar EOG channel(s) …'
            logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                        session=session))
        for eog_ch_name in config.eog_channels:
            if eog_ch_name in config.eeg_bipolar_channels:
                msg = f'    {eog_ch_name} -> eog'
                logger.info(gen_log_message(message=msg, step=0,
                                            subject=subject,
                                            session=session))
                raw.set_channel_types({eog_ch_name: 'eog'})


def set_eeg_montage(raw, subject, session) -> None:
    """Set an EEG template montage if requested.

    Modifies ``raw`` in-place.
    """
    montage_name = config.eeg_template_montage
    if config.get_datatype() == 'eeg' and montage_name:
        msg = (f'Setting EEG channel locations to template montage: '
               f'{montage_name}.')
        logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                    session=session))
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False, on_missing='warn')


def fix_stim_artifact(raw: mne.io.BaseRaw) -> None:
    """Fix stimulation artifact in the data."""
    if not config.fix_stim_artifact:
        return

    events, _ = mne.events_from_annotations(raw)
    mne.preprocessing.fix_stim_artifact(
        raw, events=events, event_id=None,
        tmin=config.stim_artifact_tmin,
        tmax=config.stim_artifact_tmax,
        mode='linear')


def import_experimental_data(
    *,
    subject: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    save: bool = False
) -> mne.io.BaseRaw:
    """Run the data import.

    Parameters
    ----------
    subject
        The subject to import.
    session
        The session to import.
    run
        The run to import.
    save
        Whether to save the data to disk or not.
    """
    bids_path_in = BIDSPath(subject=subject,
                            session=session,
                            run=run,
                            task=config.get_task(),
                            acquisition=config.acq,
                            processing=config.proc,
                            recording=config.rec,
                            space=config.space,
                            suffix=config.get_datatype(),
                            datatype=config.get_datatype(),
                            root=config.get_bids_root())
    bids_path_out = bids_path_in.copy().update(suffix='raw',
                                               extension='.fif',
                                               root=config.get_deriv_root(),
                                               check=False)

    raw = load_data(bids_path_in)
    set_eeg_montage(raw=raw, subject=subject, session=session)
    create_bipolar_channels(raw=raw, subject=subject, session=session)
    drop_channels(raw=raw, subject=subject, session=session)
    rename_events(raw=raw, subject=subject, session=session)
    fix_stim_artifact(raw=raw)
    find_bad_channels(raw=raw, subject=subject, session=session,
                      task=config.get_task(), run=run)

    # Save the data.
    if save:
        raw.save(fname=bids_path_out, split_naming='bids', overwrite=True)

    return raw


def import_er_data(
    *,
    subject: str,
    session: Optional[str] = None,
    bads: List[str],
    save: bool = False
) -> mne.io.BaseRaw:
    """Import empty-room data.

    Parameters
    ----------
    subject
        The subject for whom to import the empty-room data.
    session
        The session for which to import the empty-room data.
    bads
        The selection of bad channels from the corresponding experimental
        recording.
    save
        Whether to save the data to disk or not.
    """
    bids_path_er_in = BIDSPath(
        subject=subject,
        session=session,
        run=config.get_runs(subject=subject)[0],
        task=config.get_task(),
        acquisition=config.acq,
        processing=config.proc,
        recording=config.rec,
        space=config.space,
        suffix=config.get_datatype(),
        datatype=config.get_datatype(),
        root=config.get_bids_root()
    ).find_empty_room()

    bids_path_er_out = (bids_path_er_in.copy()
                        .update(task='noise',
                                run=None,
                                suffix='raw',
                                extension='.fif',
                                root=config.get_deriv_root(),
                                check=False))

    if bads is None:
        bads = []

    raw_er = load_data(bids_path_er_in)
    drop_channels(raw=raw_er, subject='emptyroom', session=session)

    # Set same set of bads as in the experimental run, but only for MEG
    # channels (because we won't have any others in empty-room recordings)
    raw_er.info['bads'] = [ch for ch in bads if ch.startswith('MEG')]

    # Save the data.
    if save:
        raw_er.save(fname=bids_path_er_out, split_naming='bids',
                    overwrite=True)

    return raw_er


def get_reference_run_info(
    *,
    subject: str,
    session: Optional[str] = None,
    run: str
) -> mne.Info:

    msg = f'Loading info for run: {run}.'
    logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                session=session))

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        run=run,
        task=config.get_task(),
        acquisition=config.acq,
        recording=config.rec,
        space=config.space,
        suffix='meg',
        extension='.fif',
        datatype=config.get_datatype(),
        root=config.get_bids_root(),
    )

    info = mne.io.read_info(bids_path)
    return info


exports = [
    import_experimental_data,
    import_er_data,
    get_reference_run_info
]
