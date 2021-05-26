"""
====================================================================
01. Import data, add or remove channels, detect bads, rename events.
====================================================================

The data are imported from the BIDS folder.

Notes
-----
This is the first step of the pipeline, so it will also write a
`dataset_description.json` file to the root of the pipeline derivatives, which
are stored in bids_root/derivatives/PIPELINE_NAME. PIPELINE_NAME is defined in
the config.py file. The `dataset_description.json` file is formatted according
to the WIP specification for common BIDS derivatives, see this PR:

https://github.com/bids-standard/bids-specification/pull/265
"""  # noqa: E501

import itertools
import logging

import numpy as np
import pandas as pd
import json_tricks

import mne
from mne.preprocessing import find_bad_channels_maxwell
from mne.parallel import parallel_func
from mne_bids import BIDSPath, read_raw_bids

import config
from config import (gen_log_message, on_error, failsafe_run, get_mf_cal_fname,
                    get_mf_ctc_fname)

logger = logging.getLogger('mne-bids-pipeline')


def rename_events(raw, subject, session) -> None:
    """Rename events (actually, annotations descriptions) in ``raw``.

    Modifies ``raw`` in-place.
    """
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
    session = bids_path.session

    raw = read_raw_bids(bids_path=bids_path)

    crop_data(raw, subject, session)

    raw.load_data()
    if hasattr(raw, 'fix_mag_coil_types'):
        raw.fix_mag_coil_types()

    set_eeg_montage(raw, subject, session)
    create_bipolar_channels(raw, subject, session)
    drop_channels(raw, subject, session)

    return raw


def crop_data(raw, subject, session):
    """Crop the data to the desired duration.

    Modifies ``raw`` in-place.
    """
    if subject != 'emptyroom' and config.crop is not None:
        raw.crop(*config.crop)


def drop_channels(subject, session, raw) -> None:
    """Drop channels from the data.

    Modifies ``raw`` in-place.
    """
    if config.drop_channels:
        msg = f'Dropping channels: {", ".join(config.drop_channels)}'
        logger.info(gen_log_message(message=msg, step=0, subject=subject,
                                    session=session))
        raw.drop_channels(config.drop_channels)


def create_bipolar_channels(raw, subject, session,) -> None:
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


def fix_stim_artifact(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Fix stimulation artifact in the data."""
    events, _ = mne.events_from_annotations(raw)
    raw = mne.preprocessing.fix_stim_artifact(
        raw, events=events, event_id=None,
        tmin=config.stim_artifact_tmin,
        tmax=config.stim_artifact_tmax,
        mode='linear')
    return raw


def run_import(subject, session=None):
    bids_path_in = BIDSPath(subject=subject,
                            session=session,
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

    for run_idx, run in enumerate(config.get_runs()):
        bids_path_in.update(run=run)
        bids_path_out.update(run=run)
        raw = load_data(bids_path_in)

        # Fix stimulation artifact
        if config.fix_stim_artifact:
            raw = fix_stim_artifact(raw)

        # Auto-detect bad channels.
        if config.find_flat_channels_meg or config.find_noisy_channels_meg:
            find_bad_channels(raw=raw, subject=subject, session=session,
                              task=config.get_task(), run=run)

        # Save the data.
        kwargs = dict(fname=bids_path_out, split_naming='bids', overwrite=True)
        if not config.use_maxwell_filter:
            # Save only the channel types we wish to analyze (including the
            # channels marked as "bad").
            # We do not run `raw.pick()` here because it uses too much memory.
            # Note that we skip this bit if `use_maxwell_filter=True`, as
            # we'll want to retain all channels for Maxwell filtering in that
            # case. The Maxwell filtering script will do this channel type
            # subsetting after Maxwell filtering is complete.
            kwargs['picks'] = config.get_channels_to_analyze(raw.info)

        raw.save(**kwargs)

        # Empty-room processing.
        if run_idx == 0 and config.process_er:
            msg = 'Importing empty-room recording …'
            logger.info(gen_log_message(step=0, subject=subject,
                                        session=session, message=msg))

            bids_path_er_in = bids_path_in.find_empty_room()
            raw_er = load_data(bids_path_er_in)
            raw_er.info['bads'] = [ch for ch in raw.info['bads'] if
                                   ch.startswith('MEG')]

            raw_er_fname_out = bids_path_out.copy().update(task='noise',
                                                           run=None)

            kwargs = dict(fname=raw_er_fname_out, split_naming='bids',
                          overwrite=True)
            if not config.use_maxwell_filter:
                # Save only the channel types we wish to analyze
                # (same as for experimental data above).
                kwargs['picks'] = config.get_channels_to_analyze(raw_er.info)

            raw_er.save(**kwargs)
            del raw_er


@failsafe_run(on_error=on_error)
def main():
    """Import the data."""
    msg = 'Running Step: Data import'
    logger.info(gen_log_message(step=0, message=msg))

    ch_types = config.ch_types
    if (config.use_maxwell_filter or
            config.rename_events or
            config.drop_channels or
            (ch_types != ['eeg'] and config.find_flat_channels_meg) or
            (ch_types != ['eeg'] and config.find_noisy_channels_meg) or
            (ch_types == ['eeg'] and config.eeg_bipolar_channels)):
        parallel, run_func, _ = parallel_func(run_import, n_jobs=config.N_JOBS)
        parallel(run_func(subject, session) for subject, session in
                 itertools.product(config.get_subjects(),
                                   config.get_sessions()))

    msg = 'Completed Step: Data import'
    logger.info(gen_log_message(step=0, message=msg))


if __name__ == '__main__':
    main()
