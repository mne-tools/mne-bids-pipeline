"""
===============
05b. Apply SSP
===============

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def apply_ssp(subject, session=None):
    # load epochs to reject ICA components
    # compute SSP on first run of raw

    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root)

    fname_in = bids_path.copy().update(suffix='epo', check=False)
    fname_out = bids_path.copy().update(processing='clean', suffix='epo',
                                        check=False)

    epochs = mne.read_epochs(fname_in, preload=True)

    msg = f'Input: {fname_in}, Output: {fname_out}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    proj_fname_in = bids_path.copy().update(suffix='proj', check=False)

    msg = f'Reading SSP projections from : {proj_fname_in}'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))

    projs = mne.read_proj(proj_fname_in)
    epochs_cleaned = epochs.copy().add_proj(projs).apply_proj()
    epochs_cleaned.apply_baseline(config.baseline)

    msg = 'Saving epochs'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned.save(fname_out, overwrite=True)


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
