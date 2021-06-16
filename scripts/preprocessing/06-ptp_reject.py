"""
========================================================
06. Remove epochs based on peak-to-peak (PTP) amplitudes
========================================================

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


def drop_ptp(cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix='epo',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    infile_processing = cfg.spatial_filter
    fname_in = bids_path.copy().update(processing=infile_processing)
    fname_out = bids_path.copy().update(processing='clean')

    msg = f'Input: {fname_in}, Output: {fname_out}'
    logger.info(gen_log_message(message=msg, step=6, subject=subject,
                                session=session))

    # Get rejection parameters and drop bad epochs
    epochs = mne.read_epochs(fname_in, preload=True)
    n_epochs_before_reject = len(epochs)
    epochs.reject_tmin = cfg.reject_tmin
    epochs.reject_tmax = cfg.reject_tmax
    epochs.drop_bad(reject=cfg.reject)
    n_epochs_after_reject = len(epochs)

    if 0 < n_epochs_after_reject < 0.5 * n_epochs_before_reject:
        msg = ('More than 50% of all epochs rejected. Please check the '
               'rejection thresholds.')
        logger.warning(gen_log_message(message=msg, step=6, subject=subject,
                                       session=session))
    elif n_epochs_after_reject == 0:
        raise RuntimeError('No epochs remaining after peak-to-peak-based '
                           'rejection. Cannot continue.')

    logger.info(gen_log_message(message=msg, step=6, subject=subject,
                                session=session))
    msg = 'Saving cleaned, baseline-corrected epochs …'

    epochs.apply_baseline(cfg.baseline)
    epochs.save(fname_out, overwrite=True)


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
        baseline=config.baseline,
        reject=config.get_reject(),
        reject_tmin=config.reject_tmin,
        reject_tmax=config.reject_tmax,
        spatial_filter=config.spatial_filter,
        deriv_root=config.get_deriv_root(),
    )
    return cfg


@failsafe_run(on_error=on_error)
def main():
    """Run epochs."""
    msg = 'Running Step 6: Reject epochs based on peak-to-peak amplitude'
    logger.info(gen_log_message(step=6, message=msg))

    parallel, run_func, _ = parallel_func(drop_ptp, n_jobs=config.N_JOBS)
    parallel(run_func(get_config(), subject, session) for subject, session in
             itertools.product(config.get_subjects(),
                               config.get_sessions()))

    msg = 'Completed Step 6: Reject epochs based on peak-to-peak amplitude'
    logger.info(gen_log_message(step=6, message=msg))


if __name__ == '__main__':
    main()
