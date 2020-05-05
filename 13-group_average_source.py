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

from mne_bids import make_bids_basename

import config


def morph_stc(subject, session=None):
    print("Processing subject: %s" % subject)
    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.get_kind())

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    mne.utils.set_config('SUBJECTS_DIR', config.get_subjects_dir())
    mne.datasets.fetch_fsaverage(subjects_dir=config.get_subjects_dir())

    morphed_stcs = []
    for condition in config.conditions:
        method = config.inverse_method
        cond_str = 'cond-%s' % condition.replace(op.sep, '')
        inverse_str = 'inverse-%s' % method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph-fsaverage'
        fname_stc = op.join(fpath_deriv, '_'.join([bids_basename, cond_str,
                                                   inverse_str, hemi_str]))
        fname_stc_fsaverage = op.join(fpath_deriv,
                                      '_'.join([bids_basename, cond_str,
                                                inverse_str, morph_str,
                                                hemi_str]))

        stc = mne.read_source_estimate(fname_stc)
        morph = mne.compute_source_morph(
            stc, subject_from=subject, subject_to='fsaverage',
            subjects_dir=config.get_subjects_dir())
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(fname_stc_fsaverage)
        morphed_stcs.append(stc_fsaverage)

        del fname_stc, fname_stc_fsaverage

    return morphed_stcs


def main():
    """Run grp ave."""
    parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)
    all_morphed_stcs = parallel(run_func(subject, session)
                                for subject, session in
                                itertools.product(config.get_subjects(),
                                                  config.get_sessions()))
    all_morphed_stcs = [morphed_stcs for morphed_stcs, subject in
                        zip(all_morphed_stcs, config.get_subjects())]
    mean_morphed_stcs = map(sum, zip(*all_morphed_stcs))

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME)

    bids_basename = make_bids_basename(task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    for condition, this_stc in zip(config.conditions, mean_morphed_stcs):
        this_stc /= len(all_morphed_stcs)

        method = config.inverse_method
        cond_str = 'cond-%s' % condition.replace(op.sep, '')
        inverse_str = 'inverse-%s' % method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph-fsaverage'

        fname_stc_avg = op.join(fpath_deriv, '_'.join(['average',
                                                       bids_basename, cond_str,
                                                       inverse_str, morph_str,
                                                       hemi_str]))
        this_stc.save(fname_stc_avg)


if __name__ == '__main__':
    main()
