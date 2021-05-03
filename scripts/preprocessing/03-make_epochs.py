"""
====================
03. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs.
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath
from autoreject import get_rejection_threshold
import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)

def _get_global_reject_epochs(raw, events, event_id, tmin, tmax):
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
        proj=False, baseline=None, reject=None)
    epochs.load_data()
    epochs.apply_proj()
    reject = get_rejection_threshold(epochs, ch_types=['mag', 'grad'])
    return reject

def run_epochs(subject, session=None):
    """Extract epochs for one subject."""
    raw_list = list()
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root())

    for run in config.get_runs():
        # Prepare a name to save the data
        raw_fname_in = bids_path.copy().update(run=run, processing='filt',
                                               suffix='raw', check=False)

        if raw_fname_in.copy().update(split='01').fpath.exists():
            raw_fname_in.update(split='01')

        msg = f'Loading filtered raw data from {raw_fname_in}'
        logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                    session=session, run=run))

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
        raw_list.append(raw)

    msg = 'Concatenating runs'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))

    if len(raw_list) == 1:  # avoid extra memory usage
        raw = raw_list[0]
    else:
        raw = mne.concatenate_raws(raw_list)

    # Compute events for rest tasks
    if config.task == 'rest':
        stop = raw.times[-1] - config.fixed_length_epochs_duration
        duration = config.epochs_tmax - config.epochs_tmin
        assert config.epochs_tmin == 0., "epochs_tmin must be 0 for rest"
        assert config.fixed_length_epochs_overlap is not None, \
            "epochs_overlap cannot be None for rest"
        events = mne.make_fixed_length_events(
            raw, id=3000, start=0,
            duration=config.fixed_length_epochs_duration,
            overlap=config.fixed_length_epochs_overlap,
            stop=stop)
        event_id = dict(rest=3000)
    else : # Events for task runs
        events, event_id = mne.events_from_annotations(raw)

    if "eeg" in config.ch_types:
        projection = True if config.eeg_reference == 'average' else False
        raw.set_eeg_reference(config.eeg_reference, projection=projection)

    del raw_list

    # Construct metadata from the epochs
    if config.epochs_metadata_tmin is None:
        epochs_metadata_tmin = config.epochs_tmin
    else:
        epochs_metadata_tmin = config.epochs_metadata_tmin

    if config.epochs_metadata_tmax is None:
        epochs_metadata_tmax = config.epochs_tmax
    else:
        epochs_metadata_tmax = config.epochs_metadata_tmax

    metadata, _, _ = mne.epochs.make_metadata(
        events=events, event_id=event_id,
        tmin=epochs_metadata_tmin, tmax=epochs_metadata_tmax,
        keep_first=config.epochs_metadata_keep_first,
        keep_last=config.epochs_metadata_keep_last,
        sfreq=raw.info['sfreq'])

    # Epoch the data
    # Do not reject based on peak-to-peak or flatness thresholds at this stage
    msg = (f'Creating epochs with duration: '
           f'[{config.epochs_tmin}, {config.epochs_tmax}] sec')
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))

    reject = config.get_reject()
    if reject == 'auto':
        msg = "Using AutoReject to estimate reject parameter"
        logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                    session=session))
        reject = _get_global_reject_epochs(
            raw, tmin=config.epochs_tmin,
            tmax=config.epochs_tmax,
            events=events,
            event_id=event_id
        )
        msg = f"reject = {reject}"
        logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                    session=session))

    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=config.epochs_tmin, tmax=config.epochs_tmax,
                        proj=True, baseline=None,
                        preload=False, decim=config.decim,
                        metadata=metadata,
                        event_repeated=config.event_repeated)

    msg = 'Writing epochs to disk'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))
    epochs_fname = bids_path.copy().update(suffix='epo', check=False)
    epochs.save(epochs_fname, overwrite=True)

    if config.interactive:
        epochs.plot()
        epochs.plot_image(combine='gfp', sigma=2., cmap='YlGnBu_r')


def main():
    """Run epochs."""
    msg = 'Running Step 3: Epoching'
    logger.info(gen_log_message(step=3, message=msg))

    # Here we use fewer N_JOBS to prevent potential memory problems
    parallel, run_func, _ = parallel_func(run_epochs,
                                          n_jobs=max(config.N_JOBS // 4, 1))
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 3: Epoching'
    logger.info(gen_log_message(step=3, message=msg))


if __name__ == '__main__':
    main()
