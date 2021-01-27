"""
===========
04. Run SSP
===========

Compute Signal Suspace Projections (SSP).
"""

import itertools
import logging

import mne
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
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
                         root=config.deriv_root)

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
    msg = 'Computing SSPs for ECG'
    logger.debug(gen_log_message(message=msg, step=4, subject=subject,
                                 session=session))
    ecg_projs, ecg_events = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0,
                                             average=True)
    msg = 'Computing SSPs for EOG'
    logger.debug(gen_log_message(message=msg, step=4, subject=subject,
                                 session=session))
    if config.eog_channels:
        assert all([ch_name in raw.ch_names
                    for ch_name in config.eog_channels])
        ch_name = ','.join(config.eog_channels)
    else:
        ch_name = None

    eog_projs, eog_events = compute_proj_eog(raw, ch_name=ch_name,
                                             n_grad=1, n_mag=1, n_eeg=1,
                                             average=True)

    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


def main():
    """Run SSP."""
    if not config.use_ssp:
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
