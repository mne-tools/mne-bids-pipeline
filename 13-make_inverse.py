"""
====================
13. Inverse solution
====================

Compute and apply a dSPM inverse solution for each evoked data set.
"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import make_bids_basename

import config


def run_inverse(subject):
    print("Processing subject: %s" % subject)

    subject_path = op.join('sub-{}'.format(subject), config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=config.ses,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=config.run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    fpath_deriv = op.join(config.bids_root, 'derivatives', subject_path)
    fname_ave = \
        op.join(fpath_deriv, bids_basename + '-ave.fif')

    fname_fwd = \
        op.join(fpath_deriv, bids_basename + '-fwd.fif')

    fname_cov = \
        op.join(fpath_deriv, bids_basename + '-cov.fif')

    fname_inv = \
        op.join(fpath_deriv, bids_basename + '-inv.fif')

    evokeds = mne.read_evokeds(fname_ave)
    cov = mne.read_cov(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    info = evokeds[0].info
    inverse_operator = make_inverse_operator(
        info, forward, cov, loose=0.2, depth=0.8)
    write_inverse_operator(fname_inv, inverse_operator)

    # Apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    for condition, evoked in zip(config.conditions, evokeds):
        stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM",
                            pick_ori=None)
        stc.save(op.join(fpath_deriv, '%s_%s_mne_dSPM_inverse-%s'
                         % (config.study_name, subject,
                            condition.replace(op.sep, ''))))


parallel, run_func, _ = parallel_func(run_inverse, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
