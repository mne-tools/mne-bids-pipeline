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

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)


###############################################################################
def run_epochs(subject, session=None):
    """Extract epochs for one subject."""
    print("Processing subject: %s" % subject)

    raw_list = list()
    print("  Loading raw data")

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    for run_idx, run in enumerate(config.get_runs()):

        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.task,
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )
        # Prepare a name to save the data
        fpath_deriv = op.join(config.bids_root, 'derivatives',
                              config.PIPELINE_NAME, subject_path)

        raw_fname_in = \
            op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

        print("Input: ", raw_fname_in)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
        raw_list.append(raw)

    print('  Concatenating runs')
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

    eeg = 'eeg' in config.ch_types or config.kind == 'eeg'

    picks = mne.pick_types(raw.info, meg=meg, eeg=eeg, stim=True,
                           eog=True, exclude=())

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object:
    # https://martinos.org/mne/dev/auto_tutorials/plot_metadata_epochs.html

    # Epoch the data
    print('  Epoching')
    epochs = mne.Epochs(raw, events, event_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=config.baseline,
                        preload=False, decim=config.decim,
                        reject=config.reject)

    print('  Writing epochs to disk')
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    epochs_fname = \
        op.join(fpath_deriv, bids_basename + '-epo.fif')
    epochs.save(epochs_fname, overwrite=True)

    if config.plot:
        epochs.plot()
        epochs.plot_image(combine='gfp', picks=config.ch_types, sigma=2.,
                          cmap='YlGnBu_r')


def main():
    """Run epochs."""
    # Here we use fewer N_JOBS to prevent potential memory problems
    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))


if __name__ == '__main__':
    main()
