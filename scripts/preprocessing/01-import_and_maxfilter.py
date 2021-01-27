"""
========================
01. Apply Maxwell filter
========================

The data are imported from the BIDS folder.

If you chose to run Maxwell filter (config.use_maxwell_filter = True),
the data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files.

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
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def get_mf_cal_fname(subject, session):
    mf_cal_fpath = BIDSPath(subject=subject,
                            session=session,
                            suffix='meg',
                            datatype='meg',
                            root=config.bids_root).meg_calibration_fpath
    if mf_cal_fpath is None:
        raise ValueError('Could not find Maxwell Filter Calibration file.')

    return mf_cal_fpath


def get_mf_ctc_fname(subject, session):
    mf_ctc_fpath = BIDSPath(subject=subject,
                            session=session,
                            suffix='meg',
                            datatype='meg',
                            root=config.bids_root).meg_crosstalk_fpath
    if mf_ctc_fpath is None:
        raise ValueError('Could not find Maxwell Filter cross-talk file.')

    return mf_ctc_fpath


def rename_events(raw, subject, session):
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
            logger.warn(msg)
        else:
            raise ValueError(msg)

    # Do the actual event renaming.
    msg = 'Renaming events …'
    logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                session=session))
    descriptions = list(raw.annotations.description)
    for old_event_name, new_event_name in config.rename_events.items():
        msg = f'… {old_event_name} -> {new_event_name}'
        logger.info(gen_log_message(message=msg, step=1,
                                    subject=subject, session=session))
        for idx, description in enumerate(descriptions.copy()):
            if description == old_event_name:
                descriptions[idx] = new_event_name

    descriptions = np.asarray(descriptions, dtype=str)
    raw.annotations.description = descriptions


def find_bad_channels(raw, subject, session, task, run):
    if (config.find_flat_channels_meg and
            not config.find_noisy_channels_meg):
        msg = 'Finding flat channels.'
    elif (config.find_noisy_channels_meg and
            not config.find_flat_channels_meg):
        msg = 'Finding noisy channels using Maxwell filtering.'
    else:
        msg = ('Finding flat channels, and noisy channels using '
               'Maxwell filtering.')

    logger.info(gen_log_message(message=msg, step=1, subject=subject,
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
                         root=config.deriv_root)

    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw=raw,
        calibration=get_mf_cal_fname(subject, session),
        cross_talk=get_mf_ctc_fname(subject, session),
        return_scores=True)

    preexisting_bads = raw.info['bads'].copy()
    bads = preexisting_bads.copy()

    if config.find_flat_channels_meg:
        msg = f'Found {len(auto_flat_chs)} flat channels.'
        logger.info(gen_log_message(message=msg, step=1,
                                    subject=subject, session=session))
        bads.extend(auto_flat_chs)
    if config.find_noisy_channels_meg:
        msg = f'Found {len(auto_noisy_chs)} noisy channels.'
        logger.info(gen_log_message(message=msg, step=1,
                                    subject=subject, session=session))
        bads.extend(auto_noisy_chs)

    bads = sorted(set(bads))
    raw.info['bads'] = bads
    msg = f'Marked {len(raw.info["bads"])} channels as bad.'
    logger.info(gen_log_message(message=msg, step=1,
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
        reasons.extend(['pre-existing (before mne-study-template was run)'] *
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

    extra_params = dict()
    if config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    raw = read_raw_bids(bids_path=bids_path,
                        extra_params=extra_params)

    if config.daysback is not None:
        raw.anonymize(daysback=config.daysback)

    if subject != 'emptyroom':
        # Crop the data.
        if config.crop is not None:
            raw.crop(*config.crop)

        # Rename events.
        if config.rename_events:
            rename_events(raw=raw, subject=subject, session=session)

    raw.load_data()
    if hasattr(raw, 'fix_mag_coil_types'):
        raw.fix_mag_coil_types()

    montage_name = config.eeg_template_montage
    if config.get_datatype() == 'eeg' and montage_name:
        msg = (f'Setting EEG channel locatiions to template montage: '
               f'{montage_name}.')
        logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                    session=session))
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False, on_missing='warn')

    if config.ch_types == ['eeg'] and config.eeg_bipolar_channels:
        msg = 'Creating bipolar channels …'
        logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                    session=session))
        raw.load_data()
        for anode, cathode, ch_name in config.eeg_bipolar_channels:
            msg = f'    {anode} – {cathode} -> {ch_name}'
            logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                        session=session))
            mne.set_bipolar_reference(raw, anode=anode, cathode=cathode,
                                      ch_name=ch_name, drop_refs=False,
                                      copy=False)

    if config.drop_channels:
        msg = f'Dropping channels: {", ".join(config.drop_channels)}'
        logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                    session=session))
        raw.drop_channels(config.drop_channels)

    return raw


def run_maxwell_filter(subject, session=None):
    if config.proc and 'sss' in config.proc and config.use_maxwell_filter:
        raise ValueError(f'You cannot set use_maxwell_filter to True '
                         f'if data have already processed with Maxwell-filter.'
                         f' Got proc={config.proc}.')

    bids_path_in = BIDSPath(subject=subject,
                            session=session,
                            task=config.get_task(),
                            acquisition=config.acq,
                            processing=config.proc,
                            recording=config.rec,
                            space=config.space,
                            suffix=config.get_datatype(),
                            datatype=config.get_datatype(),
                            root=config.bids_root)
    bids_path_out = bids_path_in.copy().update(suffix='raw',
                                               root=config.deriv_root,
                                               check=False)

    # Load dev_head_t and digitization points from reference run.
    # Re-use in all runs and for processing empty-room recording.
    reference_run = config.get_mf_reference_run()
    # XXX Loading info would suffice!
    bids_path_in.update(run=reference_run)
    raw = load_data(bids_path_in)
    dev_head_t = raw.info['dev_head_t']
    dig = raw.info['dig']
    del reference_run, raw

    for run_idx, run in enumerate(config.get_runs()):
        bids_path_in.update(run=run)
        bids_path_out.update(run=run)
        raw = load_data(bids_path_in)

        # Fix stimulation artifact
        if config.fix_stim_artifact:
            events, _ = mne.events_from_annotations(raw)
            raw = mne.preprocessing.fix_stim_artifact(
                raw, events=events, event_id=None,
                tmin=config.stim_artifact_tmin,
                tmax=config.stim_artifact_tmax,
                mode='linear')

        # Auto-detect bad channels.
        if config.find_flat_channels_meg or config.find_noisy_channels_meg:
            find_bad_channels(raw=raw, subject=subject, session=session,
                              task=config.get_task(), run=run)

        # Maxwell-filter experimental data.
        if config.use_maxwell_filter:
            msg = 'Applying Maxwell filter to experimental data.'
            logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                        session=session))

            # Warn if no bad channels are set before Maxwell filter
            if not raw.info['bads']:
                msg = '\nFound no bad channels. \n '
                logger.warning(gen_log_message(message=msg, subject=subject,
                                               step=1, session=session))

            if config.mf_st_duration:
                msg = '    st_duration=%d' % (config.mf_st_duration)
                logger.info(gen_log_message(message=msg, step=1,
                                            subject=subject, session=session))

            # Keyword arguments shared between Maxwell filtering of the
            # experimental and the empty-room data.
            common_mf_kws = dict(
                calibration=get_mf_cal_fname(subject, session),
                cross_talk=get_mf_ctc_fname(subject, session),
                st_duration=config.mf_st_duration,
                origin=config.mf_head_origin,
                coord_frame='head',
                destination=dev_head_t
            )

            raw_sss = mne.preprocessing.maxwell_filter(raw, **common_mf_kws)
            raw_out = raw_sss
            raw_fname_out = (bids_path_out.copy()
                             .update(processing='sss',
                                     extension='.fif'))
        elif config.ch_types == ['eeg']:
            msg = 'Not applying Maxwell filter to EEG data.'
            logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                        session=session))
            raw_out = raw
            raw_fname_out = bids_path_out.copy().update(extension='.fif')
        else:
            msg = ('Not applying Maxwell filter.\nIf you wish to apply it, '
                   'set use_maxwell_filter=True in your configuration.')
            logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                        session=session))
            raw_out = raw
            raw_fname_out = bids_path_out.copy().update(extension='.fif')

        # Save only the channel types we wish to analyze (including the
        # channels marked as "bad").
        # We do not run `raw_out.pick()` here because it uses too much memory.
        chs_to_include = config.get_channels_to_analyze(raw_out.info)
        raw_out.save(raw_fname_out, picks=chs_to_include, overwrite=True,
                     split_naming='bids')
        del raw_out
        if config.interactive:
            # Load the data we have just written, because it contains only
            # the relevant channels.
            raw = mne.io.read_raw_fif(raw_fname_out, allow_maxshield=True)
            raw.plot(n_channels=50, butterfly=True)

        # Empty-room processing.
        #
        # We pick the empty-room recording closest in time to the first run
        # of the experimental session.
        if run_idx == 0 and config.process_er:
            msg = 'Processing empty-room recording …'
            logger.info(gen_log_message(step=1, subject=subject,
                                        session=session, message=msg))

            bids_path_er_in = bids_path_in.find_empty_room()
            raw_er = load_data(bids_path_er_in)
            raw_er.info['bads'] = [ch for ch in raw.info['bads'] if
                                   ch.startswith('MEG')]

            # Maxwell-filter empty-room data.
            if config.use_maxwell_filter:
                msg = 'Applying Maxwell filter to empty-room recording'
                logger.info(gen_log_message(message=msg, step=1,
                                            subject=subject, session=session))

                # We want to ensure we use the same coordinate frame origin in
                # empty-room and experimental data processing. To do this, we
                # inject the sensor locations and the head <> device transform
                # into the empty-room recording's info, and leave all other
                # parameters the same as for the experimental data. This is not
                # very clean, as we normally should not alter info manually,
                # except for info['bads']. Will need improvement upstream in
                # MNE-Python.
                raw_er.info['dig'] = dig
                raw_er.info['dev_head_t'] = dev_head_t
                raw_er_sss = mne.preprocessing.maxwell_filter(raw_er,
                                                              **common_mf_kws)

                # Perform a sanity check: empty-room rank should match the
                # experimental data rank after Maxwell filtering.
                rank_exp = mne.compute_rank(raw, rank='info')['meg']
                rank_er = mne.compute_rank(raw_er, rank='info')['meg']
                if not np.isclose(rank_exp, rank_er):
                    msg = (f'Experimental data rank {rank_exp:.1f} does not '
                           f'match empty-room data rank {rank_er:.1f} after '
                           f'Maxwell filtering. This indicates that the data '
                           f'were processed  differenlty.')
                    raise RuntimeError(msg)

                raw_er_out = raw_er_sss
                raw_er_fname_out = bids_path_out.copy().update(
                    processing='sss')
            else:
                raw_er_out = raw_er
                raw_er_fname_out = bids_path_out.copy()

            raw_er_fname_out = raw_er_fname_out.update(
                task='noise', extension='.fif', run=None)

            # Save only the channel types we wish to analyze
            # (same as for experimental data above).
            raw_er_out.save(raw_er_fname_out, picks=chs_to_include,
                            overwrite=True, split_naming='bids')
            del raw_er_out


@failsafe_run(on_error=on_error)
def main():
    """Run maxwell_filter."""
    msg = 'Running Step 1: Data import and Maxwell filtering'
    logger.info(gen_log_message(step=1, message=msg))

    parallel, run_func, _ = parallel_func(run_maxwell_filter,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 1: Data import and Maxwell filtering'
    logger.info(gen_log_message(step=1, message=msg))


if __name__ == '__main__':
    main()
