"""
===========
05. Run ICA
===========
This fits ICA on epoched data filtered with 1 Hz highpass, 
for this purpose only using fastICA. Separate ICAs are fitted and stored for 
MEG and EEG data. 
To actually remove designated ICA components from your data, you will have to 
run 06a-apply_ica.py. 
# XXX 06a-apply_ica.py has to be added
"""

import os.path as op

import mne
from mne.preprocessing import ICA
from mne.parallel import parallel_func

import config


decim = 11  # do not touch this value unless you know what you are doing


def run_ica(subject, tsss=config.mf_st_duration):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_list = list()
    events_list = list()
    print("  Loading raw data")

    for run in config.runs:
        run += '_filt_sss'
        raw_fname = op.join(meg_subject_dir,
                            config.base_raw_fname.format(**locals()))
        eve_fname = op.splitext(raw_fname)[0] + '-eve.fif'

        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        events = mne.read_events(eve_fname)
        events_list.append(events)

        raw.info['bads'] = config.bads[subject]
        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    raw.set_eeg_reference(projection=True)
    del raw_list

    # produce high-pass filtered version of the data for ICA
    epochs_for_ica = mne.Epochs(raw.copy().filter(l_freq=1., h_freq=None),
                                events, config.event_id, config.tmin,
                                config.tmax, proj=True, baseline=config.baseline,
                                preload=False, decim=config.decim,
                                reject=config.reject)

    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = dict({'meg': picks_meg, 'eeg': picks_eeg})

    for ch_type in ['meg', 'eeg']:
        print('Running ICA for ' + ch_type)

        if ch_type == 'meg':

            # XXX this does not work on epochs:
            # meg_maxfilter_rank = epochs_for_ica.copy().pick_types(meg=True).estimate_rank()
            n_components = 0.999

        elif ch_type == 'eeg':

            n_components = 0.999

        ica = ICA(method='fastica', random_state=config.random_state,
                  n_components=n_components)

        picks = all_picks[ch_type]

        ica.fit(epochs_for_ica, picks=picks, decim=decim)

        print('  Fit %d components (explaining at least %0.1f%% of the variance)'
              % (ica.n_components_, 100 * n_components))

        ica_name = op.join(meg_subject_dir,
                           '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name,
                                                        ch_type))
        ica.save(ica_name)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
