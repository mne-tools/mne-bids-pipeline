"""
===========
04. Run SSP
===========

Compute Signal Suspace Projections (SSP).
"""

import itertools
import logging

import mne
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.parallel import parallel_func
from mne_bids import BIDSPath
from autoreject import get_rejection_threshold

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)

def _get_global_reject_ssp(raw):
    if 'eog' in raw:
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    else:
        eog_epochs = []
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']  # we don't want to reject eog based on eog
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    # we will always have an ECG as long as there are magnetometers
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
        # here we want the eog
    else:
        reject_ecg = None

    if reject_eog is None and reject_ecg is not None:
        reject_eog = {k: v for k, v in reject_ecg.items() if k != 'eog'}
    return reject_eog, reject_ecg

def run_ssp(subject, session=None):
    # compute SSP on first run of raw
    run = config.get_runs()[0]
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=run,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root())

    # Prepare a name to save the data
    raw_fname_in = bids_path.copy().update(processing='filt', suffix='raw',
                                           check=False)

    # when saving proj, use run=None
    proj_fname_out = bids_path.copy().update(run=None, suffix='proj',
                                             check=False)

    msg = f'Input: {raw_fname_in}, Output: {proj_fname_out}'
    logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                session=session))

    if raw_fname_in.copy().update(split='01').fpath.exists():
        raw_fname_in.update(split='01')

    raw = mne.io.read_raw_fif(raw_fname_in)
    # XXX : n_xxx should be options in config
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)
    print(reject_eog)

    msg = 'Computing SSPs for ECG'
    logger.debug(gen_log_message(message=msg, step=4, subject=subject,
                                 session=session))
    ecg_projs, _ = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0,
                                    average=True)

    if not ecg_projs:
        msg = 'No ECG events could be found. No ECG projectors computed.'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

    msg = 'Computing SSPs for EOG'
    logger.debug(gen_log_message(message=msg, step=4, subject=subject,
                                 session=session))
    if config.eog_channels:
        ch_names = config.eog_channels
        assert all([ch_name in raw.ch_names
                    for ch_name in ch_names])
    else:
        ch_names = None

    eog_projs, _ = compute_proj_eog(raw, ch_name=ch_names,
                                    n_grad=1, n_mag=1, n_eeg=1,
                                    average=True)

    if not eog_projs:
        msg = 'No EOG events could be found. No EOG projectors computed.'
        logger.info(gen_log_message(message=msg, step=4, subject=subject,
                                    session=session))

    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


def main():
    """Run SSP."""
    if not config.spatial_filter == 'ssp':
        return

    msg = 'Running Step 4: SSP'
    logger.info(gen_log_message(step=4, message=msg))

    parallel, run_func, _ = parallel_func(run_ssp, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 4: SSP'
    logger.info(gen_log_message(step=4, message=msg))


if __name__ == '__main__':
    main()
