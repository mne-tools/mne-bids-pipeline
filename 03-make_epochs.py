"""
====================
04. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs.
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import os.path as op
import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


###############################################################################
@failsafe_run(on_error=on_error)
def run_epochs(subject, session=None):
    """Extract epochs for one subject."""
    raw_list = list()

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    for run in config.get_runs():
        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.get_task(),
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )
        # Prepare a name to save the data
        raw_fname_in = \
            op.join(deriv_path, bids_basename + '_filt_raw.fif')

        msg = f'Loading filtered raw data from {raw_fname_in}'
        logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                    session=session, run=run))

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
        raw_list.append(raw)

    msg = 'Concatenating runs'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))
    raw = mne.concatenate_raws(raw_list)

    events, event_id = mne.events_from_annotations(raw)
    if "eeg" in config.ch_types:
        raw.set_eeg_reference(projection=True)

    del raw_list

    meg = False
    if 'meg' in config.ch_types:
        meg = True
    elif 'grad' in config.ch_types:
        meg = 'grad'
    elif 'mag' in config.ch_types:
        meg = 'mag'

    eeg = config.get_kind() == 'eeg'

    picks = mne.pick_types(raw.info, meg=meg, eeg=eeg, stim=True,
                           eog=True, exclude=())

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object:
    # https://martinos.org/mne/dev/auto_tutorials/plot_metadata_epochs.html

    # Epoch the data
    msg = 'Epoching'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))
    epochs = mne.Epochs(raw, events, event_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=config.baseline,
                        preload=False, decim=config.decim,
                        reject=config.get_reject())

    msg = 'Writing epochs to disk'
    logger.info(gen_log_message(message=msg, step=3, subject=subject,
                                session=session))
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    epochs_fname = \
        op.join(deriv_path, bids_basename + '-epo.fif')
    epochs.save(epochs_fname, overwrite=True)

    if config.plot:
        epochs.plot()
        epochs.plot_image(combine='gfp', picks=config.ch_types, sigma=2.,
                          cmap='YlGnBu_r')


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
