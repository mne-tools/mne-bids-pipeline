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
from itertools import chain
import mne
from mne.parallel import parallel_func

import config


# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)


###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_list = list()
    events_list = list()
    print("  Loading raw data")

    for run in config.runs:
        extension = run + '_sss_raw'
        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))
        eve_fname = op.splitext(raw_fname_in)[0] + '-eve.fif'
        print("Input: ", raw_fname_in, eve_fname)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        events = mne.read_events(eve_fname)
        events_list.append(events)

        # XXX mark bads from any run – is it a problem for ICA
        # if we just exclude the bads shared by all runs ?
        if run:
            bads = set(chain(*config.bads[subject].values()))
        else:
            bads = config.bads[subject]

        raw.info['bads'] = bads
        print("added bads: ", raw.info['bads'])

        raw_list.append(raw)

    print('  Concatenating runs')
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    
    if config.eeg:
        raw.set_eeg_reference(projection=True)
    
    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=config.eeg, stim=True,
                           eog=True, exclude=())

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object:
    # https://martinos.org/mne/dev/auto_tutorials/plot_metadata_epochs.html

    # Epoch the data
    print('  Epoching')
    epochs = mne.Epochs(raw, events, config.event_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=config.baseline,
                        preload=False, decim=config.decim,
                        reject=config.reject)

    print('  Writing epochs to disk')
    extension = '-epo'
    epochs_fname = op.join(meg_subject_dir,
                           config.base_fname.format(**locals()))
    print("Output: ", epochs_fname)
    epochs.save(epochs_fname)

    if config.plot:
        epochs.plot()
        epochs.plot_image(combine='gfp', group_by='type', sigma=2., cmap="YlGnBu_r")

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
