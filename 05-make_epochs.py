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

import os.path as op
import mne
from mne.parallel import parallel_func

import config


N_JOBS = max(config.N_JOBS // 4, 1)  # make less parallel runs to limit memory usage


###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_filt_sss_raw.fif' % subject)]
    eve_fnames_in = [op.join(meg_subject_dir, '%s_audvis_filt-eve.fif' % subject)]

    raw_list = list()
    events_list = list()
    print("  Loading raw data")
    for raw_fname, eve_fname in zip(raw_fnames_in, eve_fnames_in):
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        events = mne.read_events(eve_fname)
        events_list.append(events)

        raw.info['bads'] = config.bads[subject]
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
    epochs = mne.Epochs(raw, events, config.event_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=config.baseline, preload=False,
                        decim=config.decim, reject=config.reject)

    print('  Writing to disk')
    epochs.save(op.join(meg_subject_dir, '%s-epo.fif' % subject))


###############################################################################
# Let us make the script parallel across subjects

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
