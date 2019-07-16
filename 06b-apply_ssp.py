"""
===============
06b. Apply SSP
===============

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


def apply_ssp(subject):
    print("Processing subject: %s" % subject)

    # load epochs to reject ICA components
    # compute SSP on first run of raw
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
    fname_in = \
        op.join(fpath_deriv, bids_basename + '-epo.fif')

    fname_out = \
        op.join(fpath_deriv, bids_basename + '_cleaned-epo.fif')

    epochs = mne.read_epochs(fname_in, preload=True)

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    proj_fname_in = \
        op.join(fpath_deriv, bids_basename + '_ssp-proj.fif')

    print("Reading SSP projections from : %s" % proj_fname_in)

    projs = mne.read_proj(proj_fname_in)
    epochs.add_proj(projs).apply_proj()

    print('Saving epochs')
    epochs.save(fname_out, overwrite=True)


if config.use_ssp:
    parallel, run_func, _ = parallel_func(apply_ssp, n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.subjects_list)
