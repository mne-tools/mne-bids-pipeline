"""
===============
05b. Apply SSP
===============

Blinks and ECG artifacts are automatically detected and the corresponding SSP
projections components are removed from the data.

"""

import itertools
import logging
from typing import Optional

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def apply_ssp(cfg, subject, session=None):
    # load epochs to reject ICA components
    # compute SSP on first run of raw

    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root)

    fname_in = bids_path.copy().update(suffix='epo', check=False)
    fname_out = bids_path.copy().update(processing='ssp', suffix='epo',
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

    msg = 'Saving epochs with projectors.'
    logger.info(gen_log_message(message=msg, step=5, subject=subject,
                                session=session))
    epochs_cleaned.save(fname_out, overwrite=True)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
    )
    return cfg


def main():
    """Apply ssp."""
    if not config.spatial_filter == 'ssp':
        return

    msg = 'Running Step 5: Apply SSP'
    logger.info(gen_log_message(step=5, message=msg))

    parallel, run_func, _ = parallel_func(apply_ssp, n_jobs=config.N_JOBS)
    parallel(run_func(get_config(), subject, session) for subject, session in
             itertools.product(config.get_subjects(),
                               config.get_sessions()))

    msg = 'Completed Step 5: Apply SSP'
    logger.info(gen_log_message(step=5, message=msg))


if __name__ == '__main__':
    main()
