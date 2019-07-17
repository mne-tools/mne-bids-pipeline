"""
====================
06. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs.
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import os.path as op
import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)


###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject):
    print("Processing subject: %s" % subject)

    raw_list = list()
    print("  Loading raw data")

    runs = [None]  # tmp hack
    subject_path = op.join('sub-{}'.format(subject), config.kind)

    for run_idx, run in enumerate(runs):

        bids_basename = make_bids_basename(subject=subject,
                                           session=config.ses,
                                           task=config.task,
                                           acquisition=config.acq,
                                           run=config.run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )
        # Prepare a name to save the data
        fpath_deriv = op.join(config.bids_root, 'derivatives', subject_path)
        if config.use_maxwell_filter:
            raw_fname_in = \
                op.join(fpath_deriv, bids_basename + '_sss_raw.fif')
        else:
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
    epochs_fname = \
        op.join(fpath_deriv, bids_basename + '-epo.fif')
    epochs.save(epochs_fname)

    if config.plot:
        epochs.plot()
        epochs.plot_image(combine='gfp', group_by='type', sigma=2.,
                          cmap="YlGnBu_r")


# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
