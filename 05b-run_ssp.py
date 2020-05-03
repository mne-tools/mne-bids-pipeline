"""
===========
05. Run SSP
===========

Compute Signal Suspace Projections (SSP).
"""

import os.path as op
import itertools


import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from mne.preprocessing import compute_proj_ecg, compute_proj_eog


def run_ssp(subject, session=None):
    print("Processing subject: %s" % subject)

    print("  Loading one run to compute SSPs")

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    # compute SSP on first run of raw
    run = config.get_runs()[0]

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=run,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    # Prepare a name to save the data
    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)

    raw_fname_in = \
        op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

    # when saving proj, use bids_basename=None
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    proj_fname_out = op.join(fpath_deriv, bids_basename + '_ssp-proj.fif')

    print("Input: ", raw_fname_in)
    print("Output: ", proj_fname_out)

    raw = mne.io.read_raw_fif(raw_fname_in)
    # XXX : n_xxx should be options in config
    print("  Computing SSPs for ECG")
    ecg_projs, ecg_events = \
        compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=True)
    print("  Computing SSPs for EOG")
    eog_projs, eog_events = \
        compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, average=True)

    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


def main():
    """Run SSP."""
    if not config.use_ssp:
        return
    parallel, run_func, _ = parallel_func(run_ssp, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))


if __name__ == '__main__':
    main()
