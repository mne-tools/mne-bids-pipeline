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

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import make_epochs, gen_log_kwargs, on_error, failsafe_run

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

    # Now, generate epochs from each individual run.
    epochs_all_runs = []
    for run, raw_fname in zip(cfg.runs, raw_fnames):
        msg = f'Loading filtered raw data from {raw_fname} and creating epochs'
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
            raw=raw,
            event_id=event_id,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            metadata_tmin=cfg.epochs_metadata_tmin,
            metadata_tmax=cfg.epochs_metadata_tmax,
            metadata_keep_first=cfg.epochs_metadata_keep_first,
            metadata_keep_last=cfg.epochs_metadata_keep_last,
            event_repeated=cfg.event_repeated,
            decim=cfg.decim
        )
        if config.conditions:
            epochs = epochs[config.conditions]
        epochs_all_runs.append(epochs)
        del raw  # free memory

    # Lastly, we can concatenate the epochs and set an EEG reference
    epochs = mne.concatenate_epochs(epochs_all_runs, on_mismatch='warn')
    if "eeg" in cfg.ch_types:
        projection = True if cfg.eeg_reference == 'average' else False
        epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

    msg = (f'Created {len(epochs)} epochs with time interval: '
           f'{epochs.tmin} – {epochs.tmax} sec')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    msg = 'Writing epochs to disk'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_fname = bids_path.copy().update(suffix='epo', check=False)
    epochs.save(epochs_fname, overwrite=True)

    if cfg.interactive:
        epochs.plot()
        epochs.plot_image(combine='gfp', sigma=2., cmap='YlGnBu_r')


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
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
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        epochs_metadata_tmin=config.epochs_metadata_tmin,
        epochs_metadata_tmax=config.epochs_metadata_tmax,
        epochs_metadata_keep_first=config.epochs_metadata_keep_first,
        epochs_metadata_keep_last=config.epochs_metadata_keep_last,
        event_repeated=config.event_repeated,
        decim=config.decim,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference()
    )
    return cfg


def main():
    """Run epochs."""
    # Here we use fewer n_jobs to prevent potential memory problems
    parallel, run_func, _ = parallel_func(
        run_epochs,
        n_jobs=max(config.get_n_jobs() // 4, 1)
    )
    logs = parallel(
        run_func(cfg=get_config(subject, session), subject=subject,
                 session=session)
        for subject, session in
        itertools.product(config.get_subjects(), config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
