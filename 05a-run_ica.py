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

# Here we always process with the 1 Hz highpass data (instead of using
# l_freq) because ICA needs a highpass.


grads_rejection_limit = 4000e-13
mags_rejection_limit = 4e-12
decim = 11  # do not touch this value unless you know what you are doing


def run_ica(subject, tsss=config.mf_st_duration):
    print("Processing subject: %s" % subject)
    
    meg_subject_dir = op.join(config.meg_dir, subject)
   
    epochs_fname = op.join(meg_subject_dir,
                            config.base_epochs_fname.format(**locals()))
    epochs_for_ICA_fname = op.splitext(epochs_fname)[0] + '_for_ICA.fif'

    epochs_for_ICA = mne.read_epochs(epochs_for_ICA_fname, preload=True)

    # SSS reduces the data rank and the noise levels, so let's include
    # components based on a higher proportion of variance explained (0.999)
    # than we would otherwise do for non-Maxwell-filtered raw data (0.98)
    n_components = 0.999  # XXX: This can bring troubles to ICA
    
    ica_name = op.join(config.meg_dir, 
                       '{0}_{1}-ica.fif'.format(subject, config.study_name))
        
    # Here we only compute ICA for MEG because we only eliminate ECG artifacts,
    # which are not prevalent in EEG (blink artifacts are, but we will remove
    # trials with blinks at the epoching stage).
    print('  Fitting ICA')
    ica = ICA(method='fastica', random_state=config.random_state,
              n_components=n_components)
    
    # XXX run ICA on MEG and EEG
    picks = mne.pick_types(epochs_for_ICA.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
    
    ica.fit(epochs_for_ICA, picks=picks, decim=decim)
    
    print('  Fit %d components (explaining at least %0.1f%% of the variance)'
          % (ica.n_components_, 100 * n_components))
    ica.save(ica_name)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
