"""
=================================
14. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def morph_stc(subject):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    morphed_stcs = []
    for condition in config.conditions:
        fname_stc = op.join(meg_subject_dir, '%s_%s_mne_dSPM_inverse-%s'
                            % (config.study_name, subject,
                               condition.replace(op.sep, '')))
        stc = mne.read_source_estimate(fname_stc)

        morph = mne.compute_source_morph(stc, subject_from=subject,
                                         subject_to='fsaverage',
                                         subjects_dir=config.subjects_dir)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(
            op.join(meg_subject_dir,
                    'mne_dSPM_inverse_fsaverage-%s' % condition))
        morphed_stcs.append(stc_fsaverage)

    return morphed_stcs


parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)
all_morphed_stcs = parallel(run_func(subject)
                            for subject in config.subjects_list)
all_morphed_stcs = [morphed_stcs for morphed_stcs, subject in
                    zip(all_morphed_stcs, config.subjects_list)
                    if subject not in config.exclude_subjects]
mean_morphed_stcs = map(sum, zip(*all_morphed_stcs))

for condition, this_stc in zip(config.conditions, mean_morphed_stcs):
    this_stc /= len(all_morphed_stcs)
    this_stc.save(op.join(config.meg_dir, 'average_dSPM-%s' % condition))
