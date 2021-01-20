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
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


###############################################################################
@failsafe_run(on_error=on_error)
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
                         root=config.deriv_root)

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

    events, event_id = mne.events_from_annotations(raw)
    if "eeg" in config.ch_types:
        projection = True if config.eeg_reference == 'average' else False
        raw.set_eeg_reference(config.eeg_reference, projection=projection)

    del raw_list

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object:
    # https://martinos.org/mne/dev/auto_tutorials/plot_metadata_epochs.html

    # Epoch the data
    msg = 'Epoching'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=config.tmin, tmax=config.tmax,
                        proj=True, baseline=config.baseline,
                        preload=False, decim=config.decim,
                        reject=config.get_reject(),
                        reject_tmin=config.reject_tmin,
                        reject_tmax=config.reject_tmax)

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
