"""
====================
06. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily. Some channels were not properly defined during acquisition, so they
are redefined before epoching. Bad EEG channels are interpolated and epochs
containing blinks are rejected. Finally
the epochs are saved to disk. To save space, the epoch data can be decimated.
"""

import mne
from mne.parallel import parallel_func

import config

N_JOBS = max(config.N_JOBS // 4, 1)  # make less parallel runs to limit memory usage


###############################################################################
# We define the events and the onset and offset of the epochs

events_id = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4,
}


###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject):
    print("Processing subject: %s" % subject)

    data_path = op.join(meg_dir, subject)

    # map to correct subject for bad channels
    mapping = map_subjects[subject_id]

    raw_list = list()
    events_list = list()
    print("  Loading raw data")
    for run in range(1, 7):
        raw = mne.io.read_raw_fif(run_fname, preload=True)

        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(op.join(data_path, 'run_%02d-eve.fif' % run))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)

        raw.info['bads'] = bads[subject_id]
        raw.interpolate_bads()
        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    raw.set_eeg_reference(projection=True)
    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           eog=True, exclude=())

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object.

    # Epoch the data
    print('  Epoching')
    epochs = mne.Epochs(raw, events, config.events_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=baseline, preload=False,
                        decim=config.decim, reject=config.reject)

    print('  Writing to disk')
    epochs.save(op.join(data_path, '%s-tsss_%d-epo.fif' % (subject, tsss)))


###############################################################################
# Let us make the script parallel across subjects

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)

subjects_iterable = [config.subjects] if isinstance(config.subjects, str) else config.subjects 
parallel(run_func(subject) for subject in subjects_iterable)