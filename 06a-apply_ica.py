"""
===============
06. Evoked data
===============

The evoked data sets are created by averaging different conditions.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_evoked(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '-epo'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))
    extension = 'cleaned-epo'
    fname_out = op.join(meg_subject_dir,
                        config.base_fname.format(**locals()))

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    epochs = mne.read_epochs(fname_in, preload=True)

   for ch_type in ['meg','eeg']:
        print(ch_type)
        picks = all_picks[ch_type]
        
        ica_name = op.join(meg_subject_dir,
                           '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name,
                                                        ch_type))
        # Load ICA  
        print('Reading ICA: ' ica_name)
        ica = read_ica(fname = ica_name)
       
        
       

parallel, run_func, _ = parallel_func(run_evoked, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
