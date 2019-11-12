"""
====================
13. Inverse solution
====================

Compute and apply a dSPM inverse solution for each evoked data set.
"""

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import make_bids_basename

import config


def run_inverse(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
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
        stc.save(op.join(fpath_deriv, '%s_mne_dSPM_inverse-%s'
                         % (bids_basename,
                            condition.replace(op.sep, ''))))
       
        
def main():
    """Run inv."""
    parallel, run_func, _ = parallel_func(run_inverse, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    main()
