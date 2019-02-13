"""
===========
05. Run ICA
===========
# XXX update
ICA decomposition using fastICA.
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

    epochs_fname = op.join(meg_subject_dir,
                           config.base_epochs_fname.format(**locals()))
    
    epochs_for_ICA_fname = op.splitext(epochs_fname)[0] + '_for_ICA.fif'

    epochs_for_ICA = mne.read_epochs(epochs_for_ICA_fname, preload=True)
    
    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(epochs_for_ICA.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(epochs_for_ICA.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = dict({'meg': picks_meg, 'eeg': picks_eeg})

    for ch_type in ['meg', 'eeg']:
        print('Running ICA for ' + ch_type)
        
        if ch_type == 'meg': 
            
            # XXX this does not work on epochs: 
            # meg_maxfilter_rank = epochs_for_ICA.copy().pick_types(meg=True).estimate_rank()
            n_components=0.999
            
        elif ch_type == 'eeg': 
            
            n_components=0.999
        
        ica = ICA(method='fastica', random_state=config.random_state,
              n_components=n_components)
        
        picks = all_picks[ch_type]

        ica.fit(epochs_for_ICA, picks=picks, decim=decim)

        print('  Fit %d components (explaining at least %0.1f%% of the variance)'
              % (ica.n_components_, 100 * n_components))

        ica_name = op.join(meg_subject_dir,
                           '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name,
                                                        ch_type))
        ica.save(ica_name)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
