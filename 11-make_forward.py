"""
====================
12. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import os.path as op
import mne

from mne.parallel import parallel_func

import config


def run_forward(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    extension = '-ave'
    fname_evoked = op.join(meg_subject_dir,
                           config.base_fname.format(**locals()))
    print("Input: ", fname_evoked)

    extension = '%s-fwd' % (config.spacing)
    fname_fwd = op.join(meg_subject_dir,
                        config.base_fname.format(**locals()))
    print("Output: ", fname_fwd)

    fname_trans = op.join(meg_subject_dir,
                          config.base_fname_trans.format(**locals()))

    src = mne.setup_source_space(subject, spacing=config.spacing,
                                 subjects_dir=config.subjects_dir,
                                 add_dist=False)

    evoked = mne.read_evokeds(fname_evoked, condition=0)

    # Here we only use 1-layer BEM because the 3-layer is unreliable
    if 'eeg' in evoked:
        fname_bem = op.join(config.subjects_dir, subject, 'bem',
                            '%s-5120-5120-5120-bem-sol.fif' % subject)
    else:
        fname_bem = op.join(config.subjects_dir, subject, 'bem',
                            '%s-5120-bem-sol.fif' % subject)

    # Because we use a 1-layer BEM, we do MEG only
    fwd = mne.make_forward_solution(evoked.info, fname_trans, src, fname_bem,
                                    mindist=config.mindist)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


parallel, run_func, _ = parallel_func(run_forward, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
