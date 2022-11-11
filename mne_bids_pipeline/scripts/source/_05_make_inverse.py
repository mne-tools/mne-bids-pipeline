"""Inverse solution.

Compute and apply an inverse solution for each evoked data set.
"""

import itertools
import pathlib
from types import SimpleNamespace

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import BIDSPath

from ..._config_utils import (
    get_noise_cov_bids_path, get_subjects, sanitize_cond_name,
    get_task, get_datatype, get_deriv_root, get_sessions)
from ..._logging import logger, gen_log_kwargs
from ..._parallel import get_parallel_backend, parallel_func
from ..._run import failsafe_run, save_logs, _sanitize_callable


def get_input_fnames_inverse(*, cfg, subject, session):
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
    in_files = dict()
    in_files['info'] = bids_path.copy().update(**cfg.source_info_path_update)
    in_files['forward'] = bids_path.copy().update(suffix='fwd')
    if cfg.noise_cov != 'ad-hoc':
        in_files['cov'] = get_noise_cov_bids_path(
            cfg=cfg,
            subject=subject,
            session=session
        )
    if 'evoked' in cfg.inverse_targets:
        in_files['evoked'] = bids_path.copy().update(suffix='ave')
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_inverse,
)
def run_inverse(*, cfg, subject, session, in_files):
    # TODO: Eventually we should maybe loop over ch_types, e.g., to create
    # MEG, EEG, and MEG+EEG inverses and STCs
    fname_fwd = in_files.pop('forward')
    out_files = dict()
    out_files['inverse'] = fname_fwd.copy().update(suffix='inv')

    info = mne.io.read_info(in_files.pop('info'))

    if cfg.noise_cov == "ad-hoc":
        cov = mne.make_ad_hoc_cov(info)
    else:
        cov = mne.read_cov(in_files.pop('cov'))

    forward = mne.read_forward_solution(fname_fwd)
    del fname_fwd
    inverse_operator = make_inverse_operator(
        info, forward, cov, loose=cfg.loose, depth=cfg.depth,
        rank='info')
    write_inverse_operator(
        out_files['inverse'], inverse_operator, overwrite=True)

    # Apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    if isinstance(cfg.conditions, dict):
        conditions = list(cfg.conditions.keys())
    else:
        conditions = cfg.conditions

    if 'evoked' in in_files:
        fname_ave = in_files.pop('evoked')
        evokeds = mne.read_evokeds(fname_ave)

        for condition, evoked in zip(conditions, evokeds):
            method = cfg.inverse_method
            pick_ori = None

            cond_str = sanitize_cond_name(condition)
            inverse_str = method
            hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
            key = f'{cond_str}+{inverse_str}+{hemi_str}'
            out_files[key] = fname_ave.copy().update(
                suffix=key, extension=None)

            if "eeg" in cfg.ch_types:
                evoked.set_eeg_reference('average', projection=True)

            stc = apply_inverse(
                evoked=evoked,
                inverse_operator=inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=pick_ori
            )
            stc.save(out_files[key], overwrite=True)
            out_files[key] = pathlib.Path(str(out_files[key]) + '-lh.stc')
    return out_files


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        proc=config.proc,
        space=config.space,
        source_info_path_update=config.source_info_path_update,
        inverse_targets=config.inverse_targets,
        ch_types=config.ch_types,
        conditions=config.conditions,
        loose=config.loose,
        depth=config.depth,
        inverse_method=config.inverse_method,
        deriv_root=get_deriv_root(config),
        noise_cov=_sanitize_callable(config.noise_cov),
    )
    return cfg


def main(*, config) -> None:
    """Run inv."""
    if not config.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config):
        parallel, run_func = parallel_func(run_inverse, config=config)
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                subject=subject,
                session=session,
            )
            for subject, session in
            itertools.product(
                get_subjects(config),
                get_sessions(config)
            )
        )
    save_logs(config=config, logs=logs)
