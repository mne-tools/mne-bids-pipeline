"""
=================================
14. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func

import config


def morph_stc(subject, session=None):
    print("Processing subject: %s" % subject)
    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)

    mne.utils.set_config('SUBJECTS_DIR', config.subjects_dir)
    mne.datasets.fetch_fsaverage(subjects_dir=config.subjects_dir)

    morphed_stcs = []
    for condition in config.conditions:
        fname_stc = op.join(fpath_deriv, '%s_%s_mne_dSPM_inverse-%s'
                            % (config.study_name, subject,
                               condition.replace(op.sep, '')))
        stc = mne.read_source_estimate(fname_stc)

        morph = mne.compute_source_morph(stc, subject_from=subject,
                                         subject_to='fsaverage',
                                         subjects_dir=config.subjects_dir)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(
            op.join(fpath_deriv,
                    'mne_dSPM_inverse_fsaverage-%s' % condition))
        morphed_stcs.append(stc_fsaverage)

    return morphed_stcs


parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)
all_morphed_stcs = parallel(run_func(subject, session)
                            for subject, session in
                            itertools.product(config.subjects_list,
                                              config.sessions))
all_morphed_stcs = [morphed_stcs for morphed_stcs, subject in
                    zip(all_morphed_stcs, config.subjects_list)
                    if subject not in config.exclude_subjects]
mean_morphed_stcs = map(sum, zip(*all_morphed_stcs))

for condition, this_stc in zip(config.conditions, mean_morphed_stcs):
    this_stc /= len(all_morphed_stcs)
    this_stc.save(op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME,
                          'average_dSPM-%s' % condition))
