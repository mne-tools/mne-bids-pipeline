"""
====================
12. Inverse solution
====================

Compute and apply an inverse solution for each evoked data set.
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run, sanitize_cond_name
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def run_inverse(*, cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    fname_info = bids_path.copy().update(**cfg.source_info_path_update)
    fname_fwd = bids_path.copy().update(suffix='fwd')
    fname_cov = bids_path.copy().update(suffix='cov')
    fname_inv = bids_path.copy().update(suffix='inv')
    fname_epo = bids_path.copy().update(processing='clean', suffix='epo')

    info = mne.io.read_info(fname_info)
    if cfg.noise_cov == "ad-hoc":
        cov = mne.make_ad_hoc_cov(info)
        rank = 'info'
    else:
        cov = mne.read_cov(fname_cov)
        epo = mne.read_epochs(fname_epo)
        rank_epo = mne.compute_rank(epo, rank='info')['meg']
        rank_cov = mne.compute_rank(cov, info=epo.info)['meg']
        rank = {'meg': min(rank_epo, rank_cov)}

        msg = f'    Exp rank: {rank_epo}, noise rank: {rank_cov}.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject))


    forward = mne.read_forward_solution(fname_fwd)
    inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2,
                                             depth=0.8, rank=rank)
    write_inverse_operator(fname_inv, inverse_operator, overwrite=True)

    # Apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions

    if 'evoked' in cfg.inverse_targets:
        fname_ave = bids_path.copy().update(suffix='ave')
        evokeds = mne.read_evokeds(fname_ave)

        for condition, evoked in zip(conditions, evokeds):
            method = cfg.inverse_method
            pick_ori = None

            cond_str = sanitize_cond_name(condition)
            inverse_str = method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
            fname_stc = bids_path.copy().update(
                suffix=f'{cond_str}+{inverse_str}+{hemi_str}',
                extension=None)

            if "eeg" in cfg.ch_types:
                evoked.set_eeg_reference('average', projection=True)

            stc = apply_inverse(
                evoked=evoked,
                inverse_operator=inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=pick_ori
            )
            stc.save(fname_stc, overwrite=True)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        source_info_path_update=config.source_info_path_update,
        inverse_targets=config.inverse_targets,
        noise_cov=config.noise_cov,
        ch_types=config.ch_types,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        deriv_root=config.get_deriv_root(),
    )
    return cfg


def main():
    """Run inv."""
    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func, _ = parallel_func(
            run_inverse,
            n_jobs=config.get_n_jobs()
        )
        logs = parallel(
            run_func(cfg=get_config(), subject=subject, session=session)
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
