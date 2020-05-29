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

The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.

Notes
-----
This is the first step of the pipeline, so it will also write a
`dataset_description.json` file to the root of the pipeline derivatives, which
are stored in bids_root/derivatives/PIPELINE_NAME. PIPELINE_NAME is defined in
the config.py file. The `dataset_description.json` file is formatted according
to the WIP specification for common BIDS derivatives, see this PR:

https://github.com/bids-standard/bids-specification/pull/265
"""  # noqa: E501

import os
import os.path as op
import itertools
import logging

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import find_bad_channels_maxwell
from mne.parallel import parallel_func
from mne_bids import make_bids_basename, read_raw_bids, get_matched_empty_room
from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json, _parse_bids_filename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def init_dataset():
    """Prepare the pipeline directory in /derivatives.
    """
    os.makedirs(config.deriv_root, exist_ok=True)

    # Write a dataset_description.json for the pipeline
    ds_json = dict()
    ds_json['Name'] = config.PIPELINE_NAME + ' outputs'
    ds_json['BIDSVersion'] = BIDS_VERSION
    ds_json['PipelineDescription'] = {
        'Name': config.PIPELINE_NAME,
        'Version': config.VERSION,
        'CodeURL': config.CODE_URL,
    }
    ds_json['SourceDatasets'] = {
        'URL': 'n/a',
    }

    fname = op.join(config.deriv_root, 'dataset_description.json')
    _write_json(fname, ds_json, overwrite=True)


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
        raise ValueError(msg)

    # Do the actual event renaming.
    msg = 'Renaming events …'
    logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                session=session))
    description = raw.annotations.description
    for old_event_name, new_event_name in config.rename_events.items():
        msg = f'… {old_event_name} -> {new_event_name}'
        logger.info(gen_log_message(message=msg, step=1,
                                    subject=subject, session=session))
        description[description == old_event_name] = new_event_name


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
    raw_lp_filtered_for_maxwell = (raw.copy()
                                   .filter(l_freq=None,
                                           h_freq=40))
    auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
        raw=raw_lp_filtered_for_maxwell,
        calibration=config.mf_cal_fname,
        cross_talk=config.mf_ctc_fname)
    del raw_lp_filtered_for_maxwell

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

    # Write the bad channels to disk.
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    bads_tsv_fname = op.join(deriv_path, f'{bids_basename}_bad_channels.tsv')
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


def load_data(bids_basename):
    # read_raw_bids automatically
    # - populates bad channels using the BIDS channels.tsv
    # - sets channels types according to BIDS channels.tsv `type` column
    # - sets raw.annotations using the BIDS events.tsv

    params = _parse_bids_filename(bids_basename, verbose=False)
    subject = params['sub']
    session = params['ses']

    extra_params = dict()
    if config.allow_maxshield:
        extra_params['allow_maxshield'] = config.allow_maxshield

    raw = read_raw_bids(bids_basename=bids_basename,
                        bids_root=config.bids_root,
                        extra_params=extra_params,
                        kind=config.get_kind())

    # XXX hack to deal with dates that fif files cannot handle
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

    return raw


def run_maxwell_filter(subject, session=None):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())
    os.makedirs(deriv_path, exist_ok=True)

    for run_idx, run in enumerate(config.get_runs()):
        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.get_task(),
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space)

        raw = load_data(bids_basename)
        if run_idx == 0:
            dev_head_t = raw.info['dev_head_t']  # Re-use in all runs.

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
                logger.warn(gen_log_message(message=msg, subject=subject,
                                            step=1, session=session))

            if config.mf_st_duration:
                msg = '    st_duration=%d' % (config.mf_st_duration)
                logger.info(gen_log_message(message=msg, step=1,
                                            subject=subject, session=session))

            # Keyword arguments shared between Maxwell filtering of the
            # experimental and the empty-room data.
            common_mf_kws = dict(calibration=config.mf_cal_fname,
                                 cross_talk=config.mf_ctc_fname,
                                 st_duration=config.mf_st_duration,
                                 origin=config.mf_head_origin,
                                 coord_frame='head',
                                 destination=dev_head_t)

            raw_sss = mne.preprocessing.maxwell_filter(raw, **common_mf_kws)
            raw_out = raw_sss
            raw_fname_out = op.join(deriv_path, f'{bids_basename}_sss_raw.fif')
        else:
            msg = ('Not applying Maxwell filter.\nIf you wish to apply it, '
                   'set use_maxwell_filter=True in your configuration.')
            logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                        session=session))
            raw_out = raw
            raw_fname_out = op.join(deriv_path,
                                    f'{bids_basename}_nosss_raw.fif')
        raw_out.save(raw_fname_out, overwrite=True)
        if config.interactive:
            raw_out.plot(n_channels=50, butterfly=True)

        # Empty-room processing.
        #
        # We pick the empty-room recording closest in time to the first run
        # of the experimental session.
        if run_idx == 0 and config.noise_cov == 'emptyroom':
            msg = 'Processing empty-room recording …'
            logger.info(gen_log_message(step=1, subject=subject,
                                        session=session, message=msg))

            bids_basename_er_in = get_matched_empty_room(
                bids_basename=bids_basename,
                bids_root=config.bids_root)
            raw_er = load_data(bids_basename_er_in)
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
                raw_er.info['dig'] = raw.info['dig']
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
                raw_er_fname_out = op.join(
                    deriv_path,
                    f'{bids_basename}_emptyroom_sss_raw.fif')
            else:
                raw_er_out = raw_er
                raw_er_fname_out = op.join(
                    deriv_path,
                    f'{bids_basename}_emptyroom_nosss_raw.fif')

            raw_er_out.save(raw_er_fname_out, overwrite=True)


@failsafe_run(on_error=on_error)
def main():
    """Run maxwell_filter."""
    msg = "Initializing dataset."
    logger.info(gen_log_message(step=1, message=msg))
    init_dataset()

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
