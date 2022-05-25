"""
====================
03. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id).
Finally the epochs are saved to disk. For the moment, no rejection is applied.
To save space, the epoch data can be decimated.
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

import config
from config import make_epochs, gen_log_kwargs, on_error, failsafe_run
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def run_epochs(*, cfg, subject, session=None):
    """Extract epochs for one subject."""
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root)

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.
    raw_fnames = []
    for run in cfg.runs:
        raw_fname_in = bids_path.copy().update(run=run, processing='filt',
                                               suffix='raw', check=False)

        if raw_fname_in.copy().update(split='01').fpath.exists():
            raw_fname_in.update(split='01')

        raw_fnames.append(raw_fname_in)

    # Generate a unique event name -> event code mapping that can be used
    # across all runs.
    if cfg.task.lower() != 'rest':
        event_name_to_code_map = config.annotations_to_events(
            raw_paths=raw_fnames)

    # Store the rank & corresponding info of the run with the smallest rank.
    # We'll later manually inject this info into concatenated epochs.
    # This is ONLY relevant for Maxwell-filtered MEG data and ensures that
    # we later don't assume a rank that is too large when whitening the data
    # or performing the inverse modeling.
    smallest_rank = None
    smallest_rank_info = None

    # Now, generate epochs from each individual run.
    for idx, (run, raw_fname) in enumerate(
        zip(cfg.runs, raw_fnames)
    ):
        msg = (f'Loading filtered raw data from {raw_fname.basename} '
               f'and creating epochs')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        # Only keep the subset of the mapping that applies to the current run
        if cfg.task.lower() == 'rest':
            event_id = None  # make_epochs takes care of it.
        else:
            event_id = event_name_to_code_map.copy()
            for event_name in event_id.copy().keys():
                if event_name not in raw.annotations.description:
                    del event_id[event_name]

        msg = 'Creating task-related epochs …'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        epochs = make_epochs(
            subject=subject,
            session=session,
            task=cfg.task,
            raw=raw,
            event_id=event_id,
            conditions=cfg.conditions,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            metadata_tmin=cfg.epochs_metadata_tmin,
            metadata_tmax=cfg.epochs_metadata_tmax,
            metadata_keep_first=cfg.epochs_metadata_keep_first,
            metadata_keep_last=cfg.epochs_metadata_keep_last,
            metadata_query=cfg.epochs_metadata_query,
            event_repeated=cfg.event_repeated,
            decim=cfg.decim
        )

        epochs.load_data()  # Remove reference to raw
        del raw  # free memory

        if idx == 0:
            epochs_all_runs = epochs
        else:
            epochs_all_runs = mne.concatenate_epochs(
                [epochs_all_runs, epochs], on_mismatch='warn'
            )

        if cfg.use_maxwell_filter:
            # Keep track of the info corresponding to the run with the smallest
            # data rank.
            new_rank = mne.compute_rank(epochs, rank='info')['meg']
            if (
                (smallest_rank is None) or
                (new_rank < smallest_rank)
            ):
                smallest_rank = new_rank
                smallest_rank_info = epochs.info.copy()

        del epochs

    # Clean up namespace
    epochs = epochs_all_runs
    del epochs_all_runs

    if cfg.use_maxwell_filter and config.noise_cov == 'rest':
        bp_raw_rest = (bids_path.copy()
                       .update(
                           run=None,
                           task='rest',
                           processing='filt',
                           suffix='raw',
                           check=False
                        ))
        raw_rest_filt = mne.io.read_raw(bp_raw_rest)
        rank_rest = mne.compute_rank(raw_rest_filt, rank='info')['meg']
        if rank_rest < smallest_rank:
            msg = (
                f'The rank of the resting state data ({rank_rest}) is smaller '
                f'than the smallest rank of the "{cfg.task}" epochs '
                f'({smallest_rank}). Replacing part of the  "info" object of '
                f'the concatenated "{cfg.task}" epochs with information from '
                f'the resting-state run.'
            )
            logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                            session=session, run='rest'))
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
        assert epochs.info['ch_names'] == smallest_rank_info['ch_names']
        with epochs.info._unlock():
            epochs.info['proc_history'] = smallest_rank_info['proc_history']
            rank_epochs_new = mne.compute_rank(epochs, rank='info')['meg']
            msg = (
                f'The rank of the "{cfg.task}" epochs is now: '
                f'{rank_epochs_new}'
            )
            logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                            session=session))

    # Set an EEG reference
    if "eeg" in cfg.ch_types:
        projection = True if cfg.eeg_reference == 'average' else False
        epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

    n_epochs_before_metadata_query = len(epochs.drop_log)

    msg = (f'Created {n_epochs_before_metadata_query} epochs with time '
           f'interval: {epochs.tmin} – {epochs.tmax} sec.\n'
           f'Selected {len(epochs)} epochs via metadata query: '
           f'{cfg.epochs_metadata_query}\n'
           f'Writing {len(epochs)} epochs to disk.')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_fname = bids_path.copy().update(suffix='epo', check=False)
    epochs.save(epochs_fname, overwrite=True, split_naming='bids')

    if cfg.interactive:
        epochs.plot()
        epochs.plot_image(combine='gfp', sigma=2., cmap='YlGnBu_r')


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        process_er=config.process_er,
        runs=config.get_runs(subject=subject),
        use_maxwell_filter=config.use_maxwell_filter,
        proc=config.proc,
        task=config.get_task(),
        datatype=config.get_datatype(),
        session=session,
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.get_bids_root(),
        deriv_root=config.get_deriv_root(),
        interactive=config.interactive,
        conditions=config.conditions,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        epochs_metadata_tmin=config.epochs_metadata_tmin,
        epochs_metadata_tmax=config.epochs_metadata_tmax,
        epochs_metadata_keep_first=config.epochs_metadata_keep_first,
        epochs_metadata_keep_last=config.epochs_metadata_keep_last,
        epochs_metadata_query=config.epochs_metadata_query,
        event_repeated=config.event_repeated,
        decim=config.decim,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference()
    )
    return cfg


def main():
    """Run epochs."""
    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_epochs)
        logs = parallel(
            run_func(
                cfg=get_config(subject, session), subject=subject,
                session=session
            )
            for subject, session in
            itertools.product(config.get_subjects(), config.get_sessions())
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
