"""
===============
06b. Apply SSP
===============

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

import os.path as op
import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def apply_ssp(subject, session=None):
    # load epochs to reject ICA components
    # compute SSP on first run of raw

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    fname_in = op.join(deriv_path,
                       bids_basename.update(suffix='epo.fif'))
    fname_out = op.join(deriv_path,
                        bids_basename.update(suffix='cleaned-epo.fif'))

    epochs = mne.read_epochs(fname_in, preload=True)

    msg = f'Input: {fname_in}, Output: {fname_out}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    proj_fname_in = op.join(deriv_path,
                            bids_basename.update(suffix='ssp-proj.fif'))

    msg = f'Reading SSP projections from : {proj_fname_in}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    projs = mne.read_proj(proj_fname_in)
    epochs.add_proj(projs).apply_proj()

    msg = 'Saving epochs'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs.save(fname_out, overwrite=True)


def main():
    """Apply ssp."""
    if not config.use_ssp:
        return

    msg = 'Running Step 5: Apply SSP'
    logger.info(gen_log_message(step=5, message=msg))

    parallel, run_func, _ = parallel_func(apply_ssp, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 5: Apply SSP'
    logger.info(gen_log_message(step=5, message=msg))


if __name__ == '__main__':
    main()
