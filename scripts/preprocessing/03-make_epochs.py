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

import config
from config import make_epochs, gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def run_epochs(subject, session=None):
    """Extract epochs for one subject."""
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root())

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.
    raw_fnames = []
    for run in config.get_runs():
        raw_fname_in = bids_path.copy().update(run=run, processing='filt',
                                               suffix='raw', check=False)

        if raw_fname_in.copy().update(split='01').fpath.exists():
            raw_fname_in.update(split='01')

        raw_fnames.append(raw_fname_in)

    # Now, generate epochs from each individual run
    epochs_all_runs = []
    for raw_fname in raw_fnames:
        msg = f'Loading filtered raw data from {raw_fname} and creating epochs'
        logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                    session=session))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        epochs = make_epochs(
            raw=raw,
            tmin=config.epochs_tmin,
            tmax=config.epochs_tmax,
            metadata_tmin=config.epochs_metadata_tmin,
            metadata_tmax=config.epochs_metadata_tmax,
            metadata_keep_first=config.epochs_metadata_keep_first,
            metadata_keep_last=config.epochs_metadata_keep_last,
            event_repeated=config.event_repeated,
            decim=config.decim
        )
        epochs_all_runs.append(epochs)
        del raw  # free memory

    # Lastly, we can concatenate the epochs and set an EEG reference
    epochs = mne.concatenate_epochs(epochs_all_runs)
    if "eeg" in config.ch_types:
        projection = True if config.eeg_reference == 'average' else False
        epochs.set_eeg_reference(config.eeg_reference, projection=projection)

    msg = (f'Created {len(epochs)} epochs with time interval: '
           f'{epochs.tmin} – {epochs.tmax} sec')
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))

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
