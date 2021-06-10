"""
===========================
01. MaxWell-filter MEG data
===========================

If you chose to run Maxwell filter (config.use_maxwell_filter = True),
the data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files.
"""  # noqa: E501

import itertools
import logging

import numpy as np

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import (gen_log_message, on_error, failsafe_run, get_mf_ctc_fname,
                    get_mf_cal_fname)

logger = logging.getLogger('mne-bids-pipeline')


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
                            suffix='raw',
                            extension='.fif',
                            datatype=config.get_datatype(),
                            root=config.get_deriv_root(),
                            check=False)
    bids_path_out = bids_path_in.copy().update(processing='sss')

    # Load dev_head_t and digitization points from MaxFilter reference run.
    # Re-use in all runs and for processing empty-room recording.
    reference_run = config.get_mf_reference_run()
    msg = f'Loading reference run: {reference_run}.'
    logger.info(gen_log_message(message=msg, step=1, subject=subject,
                                session=session))
    bids_path_in.update(run=reference_run)
    info = mne.io.read_info(bids_path_in)
    dev_head_t = info['dev_head_t']
    dig = info['dig']
    del reference_run, info

    for run_idx, run in enumerate(config.get_runs(subject=subject)):
        bids_path_in.update(run=run)
        bids_path_out.update(run=run)
        raw = mne.io.read_raw_fif(bids_path_in, allow_maxshield=True)

        # Maxwell-filter experimental data.
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

        # Save only the channel types we wish to analyze (including the
        # channels marked as "bad").
        # We do not run `raw_sss.pick()` here because it uses too much memory.
        picks = config.get_channels_to_analyze(raw.info)
        raw_sss.save(bids_path_out, picks=picks, split_naming='bids',
                     overwrite=True)
        del raw_sss

        if config.interactive:
            # Load the data we have just written, because it contains only
            # the relevant channels.
            raw = mne.io.read_raw_fif(bids_path_out, allow_maxshield=True)
            raw.plot(n_channels=50, butterfly=True)

        # Empty-room processing.
        #
        # We pick the empty-room recording closest in time to the first run
        # of the experimental session.
        if run_idx == 0 and config.process_er:
            msg = 'Processing empty-room recording …'
            logger.info(gen_log_message(step=1, subject=subject,
                                        session=session, message=msg))

            bids_path_er_in = bids_path_in.copy().update(task='noise',
                                                         run=None)
            raw_er = mne.io.read_raw_fif(bids_path_er_in, allow_maxshield=True)
            raw_er.info['bads'] = [ch for ch in raw.info['bads'] if
                                   ch.startswith('MEG')]

            # Maxwell-filter empty-room data.
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
            raw_sss = mne.io.read_raw_fif(bids_path_out)
            rank_exp = mne.compute_rank(raw_sss, rank='info')['meg']
            rank_er = mne.compute_rank(raw_er_sss, rank='info')['meg']
            if not np.isclose(rank_exp, rank_er):
                msg = (f'Experimental data rank {rank_exp:.1f} does not '
                       f'match empty-room data rank {rank_er:.1f} after '
                       f'Maxwell filtering. This indicates that the data '
                       f'were processed  differently.')
                raise RuntimeError(msg)

            raw_er_fname_out = bids_path_out.copy().update(
                processing='sss')

            raw_er_fname_out = raw_er_fname_out.update(task='noise', run=None)

            # Save only the channel types we wish to analyze
            # (same as for experimental data above).
            raw_er_sss.save(raw_er_fname_out, picks=picks,
                            overwrite=True, split_naming='bids')
            del raw_er_sss


@failsafe_run(on_error=on_error)
def main():
    """Run maxwell_filter."""
    msg = 'Running Step 1: Maxwell filter'
    logger.info(gen_log_message(step=1, message=msg))

    if config.use_maxwell_filter:
        parallel, run_func, _ = parallel_func(run_maxwell_filter,
                                              n_jobs=config.N_JOBS)
        parallel(run_func(subject, session) for subject, session in
                 itertools.product(config.get_subjects(),
                                   config.get_sessions()))

    msg = 'Completed Step 1: Maxwell filter'
    logger.info(gen_log_message(step=1, message=msg))


if __name__ == '__main__':
    main()
