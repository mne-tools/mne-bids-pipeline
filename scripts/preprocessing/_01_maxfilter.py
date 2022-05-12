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
from typing import Optional
from types import SimpleNamespace

import numpy as np
import mne
from mne_bids import BIDSPath

import config
from config import (gen_log_kwargs, on_error, failsafe_run,
                    import_experimental_data, import_er_data, import_rest_data,
                    get_reference_run_params)
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def run_maxwell_filter(*, cfg, subject, session=None, run=None):
    if cfg.proc and 'sss' in cfg.proc and cfg.use_maxwell_filter:
        raise ValueError(f'You cannot set use_maxwell_filter to True '
                         f'if data have already processed with Maxwell-filter.'
                         f' Got proc={config.proc}.')

    bids_path_out = BIDSPath(subject=subject,
                             session=session,
                             run=run,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             processing='sss',
                             recording=cfg.rec,
                             space=cfg.space,
                             suffix='raw',
                             extension='.fif',
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             check=False)

    # Load dev_head_t and digitization points from MaxFilter reference run.
    if cfg.mf_reference_run is not None:
        # Only log if we have more than just a single run
        msg = f'Loading reference run: {cfg.mf_reference_run}.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))

    reference_run_params = get_reference_run_params(
        subject=subject, session=session, run=cfg.mf_reference_run
    )
    dev_head_t = reference_run_params['dev_head_t']
    del reference_run_params

    raw = import_experimental_data(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        save=False
    )

    # Maxwell-filter experimental data.
    msg = 'Applying Maxwell filter to experimental data: '

    if cfg.mf_st_duration:
        msg += f'tSSS (st_duration: {cfg.mf_st_duration} sec).'
    else:
        msg += 'SSS.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session, run=run))

    # Warn if no bad channels are set before Maxwell filter
    if not raw.info['bads']:
        msg = ('No channels were marked as bad. Please carefully check '
               'your data to ensure this is correct; otherwise, Maxwell '
               'filtering WILL cause problems.')
        logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                        session=session, run=run))

    # Keyword arguments shared between Maxwell filtering of the
    # experimental and the empty-room data.
    common_mf_kws = dict(
        calibration=cfg.mf_cal_fname,
        cross_talk=cfg.mf_ctc_fname,
        st_duration=cfg.mf_st_duration,
        origin=cfg.mf_head_origin,
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
    del raw, raw_sss

    if cfg.interactive:
        # Load the data we have just written, because it contains only
        # the relevant channels.
        raw_sss = mne.io.read_raw_fif(bids_path_out, allow_maxshield=True)
        raw_sss.plot(n_channels=50, butterfly=True, block=True)
        del raw_sss

    # Noise data processing.
    # Only process empty-room data once – we ensure this by simply checking
    # if the current run is the reference run, and only then initiate
    # processing. No sophisticated logic behind this – it's just convenient to
    # code it this way.
    if (
        (cfg.process_er or config.noise_cov == 'rest') and
        run == cfg.mf_reference_run
    ):
        if config.noise_cov == 'rest':
            recording_type = 'resting-state'
        else:
            recording_type = 'empty-room'

        msg = f'Processing {recording_type} recording …'
        logger.info(**gen_log_kwargs(subject=subject,
                                     session=session, run=run, message=msg))

        if config.noise_cov == 'rest':
            raw_noise = import_rest_data(
                cfg=cfg,
                subject=subject,
                session=session,
                save=False
            )
        else:
            raw_noise = import_er_data(
                cfg=cfg,
                subject=subject,
                session=session,
                save=False
            )

        # Maxwell-filter noise data.
        msg = f'Applying Maxwell filter to {recording_type} recording'
        logger.info(**gen_log_kwargs(message=msg,
                                     subject=subject, session=session,
                                     run=run))
        raw_noise_sss = mne.preprocessing.maxwell_filter(
            raw_noise, **common_mf_kws
        )

        # Perform a sanity check: empty-room rank should exactly match the
        # experimental data rank after Maxwell filtering; resting-state rank
        # should be equal or be greater than experimental data rank.
        #
        # We're treating the two cases differently, because we don't
        # copy the bad channel selection from the reference run over to
        # the resting-state recording.

        raw_sss = mne.io.read_raw_fif(bids_path_out)
        rank_exp = mne.compute_rank(raw_sss, rank='info')['meg']
        rank_noise = mne.compute_rank(raw_noise_sss, rank='info')['meg']

        if config.noise_cov == 'rest':
            if rank_exp > rank_noise:
                msg = (
                    f'Resting-state rank ({rank_noise}) is lower than '
                    f'reference run data rank ({rank_exp}). We will try to '
                    f'take care of this during epoching of the experimental '
                    f'data.'
                )
                logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                                session=session))
            else:
                pass  # Should cause no problems!
        elif not np.isclose(rank_exp, rank_noise):
            msg = (f'Reference run data rank {rank_exp:.1f} does not '
                   f'match {recording_type} data rank {rank_noise:.1f} after '
                   f'Maxwell filtering. This indicates that the data '
                   f'were processed  differently.')
            raise RuntimeError(msg)

        raw_noise_fname_out = bids_path_out.copy().update(
            task='rest' if config.noise_cov == 'rest' else 'noise',
            run=None,
            processing='sss'
        )

        # Save only the channel types we wish to analyze
        # (same as for experimental data above).
        raw_noise_sss.save(
            raw_noise_fname_out, picks=picks, overwrite=True,
            split_naming='bids'
        )
        del raw_noise_sss


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        mf_cal_fname=config.get_mf_cal_fname(subject, session),
        mf_ctc_fname=config.get_mf_ctc_fname(subject, session),
        mf_st_duration=config.mf_st_duration,
        mf_head_origin=config.mf_head_origin,
        process_er=config.process_er,
        runs=config.get_runs(subject=subject),  # XXX needs to accept session!
        use_maxwell_filter=config.use_maxwell_filter,
        proc=config.proc,
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.get_bids_root(),
        deriv_root=config.get_deriv_root(),
        crop_runs=config.crop_runs,
        interactive=config.interactive,
        rename_events=config.rename_events,
        eeg_template_montage=config.eeg_template_montage,
        fix_stim_artifact=config.fix_stim_artifact,
        stim_artifact_tmin=config.stim_artifact_tmin,
        stim_artifact_tmax=config.stim_artifact_tmax,
        find_flat_channels_meg=config.find_flat_channels_meg,
        find_noisy_channels_meg=config.find_noisy_channels_meg,
        mf_reference_run=config.get_mf_reference_run(),
        drop_channels=config.drop_channels,
        find_breaks=config.find_breaks,
        min_break_duration=config.min_break_duration,
        t_break_annot_start_after_previous_event=config.t_break_annot_start_after_previous_event,  # noqa:E501
        t_break_annot_stop_before_next_event=config.t_break_annot_stop_before_next_event,  # noqa:E501
    )
    return cfg


def main():
    """Run maxwell_filter."""
    if not config.use_maxwell_filter:
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_maxwell_filter)
        logs = parallel(
            run_func(
                cfg=get_config(subject, session),
                subject=subject,
                session=session,
                run=run
            )
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
            for run in config.get_runs(subject=subject)
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
