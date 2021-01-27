"""
====================
12. Inverse solution
====================

Compute and apply an inverse solution for each evoked data set.
"""

import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def run_inverse(subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    fname_ave = bids_path.copy().update(suffix='ave')
    fname_fwd = bids_path.copy().update(suffix='fwd')
    fname_cov = bids_path.copy().update(suffix='cov')
    fname_inv = bids_path.copy().update(suffix='inv')

    evokeds = mne.read_evokeds(fname_ave)
    cov = mne.read_cov(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    info = evokeds[0].info
    inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2,
                                             depth=0.8, rank='info')
    write_inverse_operator(fname_inv, inverse_operator)

    # Apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    for condition, evoked in zip(config.conditions, evokeds):
        method = config.inverse_method
        pick_ori = None

        cond_str = config.sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        fname_stc = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{hemi_str}',
            extension=None)

        if "eeg" in config.ch_types:
            evoked.set_eeg_reference('average', projection=True)

        stc = apply_inverse(evoked=evoked,
                            inverse_operator=inverse_operator,
                            lambda2=lambda2, method=method, pick_ori=pick_ori)
        stc.save(fname_stc)


def main():
    """Run inv."""
    msg = 'Running Step 12: Compute and apply inverse solution'
    logger.info(gen_log_message(step=12, message=msg))

    parallel, run_func, _ = parallel_func(run_inverse, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 12: Compute and apply inverse solution'
    logger.info(gen_log_message(step=12, message=msg))


if __name__ == '__main__':
    main()
