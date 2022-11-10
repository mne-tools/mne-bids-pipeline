"""Run Signal Subspace Projections (SSP) for artifact correction.

These are often also referred to as PCA vectors.
"""

import itertools
from types import SimpleNamespace

import mne
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne import compute_proj_evoked, compute_proj_epochs
from mne_bids import BIDSPath
from mne.utils import _pl

from ..._config_utils import (
    get_sessions, get_runs, get_subjects, get_task, get_datatype,
    get_deriv_root,
)
from ..._logging import gen_log_kwargs, logger
from ..._run import (
    failsafe_run, _update_for_splits, auto_script_path, save_logs,
)
from ..._parallel import parallel_func, get_parallel_backend
from ..._reject import _get_reject


def get_input_fnames_run_ssp(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    bids_basename = BIDSPath(subject=subject,
                             session=session,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             recording=cfg.rec,
                             space=cfg.space,
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             extension='.fif',
                             check=False)
    in_files = dict()
    for run in cfg.runs:
        key = f'raw_run-{run}'
        in_files[key] = bids_basename.copy().update(
            run=run, processing='filt', suffix='raw')
        _update_for_splits(in_files, key, single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_ssp,
)
def run_ssp(*, cfg, subject, session, in_files):
    # compute SSP on first run of raw
    raw_fnames = [in_files.pop(f'raw_run-{run}') for run in cfg.runs]

    # when saving proj, use run=None
    out_files = dict()
    out_files['proj'] = raw_fnames[0].copy().update(
        run=None, suffix='proj', split=None, processing=None, check=False)

    msg = (f'Input{_pl(raw_fnames)} ({len(raw_fnames)}): '
           f'{raw_fnames[0].basename}{_pl(raw_fnames, pl=" ...")}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = (f'Output: {out_files["proj"].basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    raw = mne.concatenate_raws([
        mne.io.read_raw_fif(raw_fname_in) for raw_fname_in in raw_fnames])
    del raw_fnames

    projs = dict()
    proj_kinds = ('ecg', 'eog')
    rate_names = dict(ecg='heart', eog='blink')
    epochs_fun = dict(ecg=create_ecg_epochs, eog=create_eog_epochs)
    minimums = dict(ecg=cfg.min_ecg_epochs, eog=cfg.min_eog_epochs)
    rejects = dict(ecg=cfg.ssp_reject_ecg, eog=cfg.ssp_reject_eog)
    avg = dict(ecg=cfg.ecg_proj_from_average, eog=cfg.eog_proj_from_average)
    n_projs = dict(ecg=cfg.n_proj_ecg, eog=cfg.n_proj_eog)
    ch_name = dict(ecg=None, eog=None)
    if cfg.eog_channels:
        ch_name['eog'] = cfg.eog_channels
        assert all([ch_name in raw.ch_names for ch_name in ch_name['eog']])
    if cfg.ssp_meg == 'auto':
        cfg.ssp_meg = 'combined' if cfg.use_maxwell_filter else 'separate'
    for kind in proj_kinds:
        projs[kind] = []
        if not any(n_projs[kind]):
            continue
        proj_epochs = epochs_fun[kind](
            raw, ch_name=ch_name[kind], decim=cfg.decim)
        n_orig = len(proj_epochs)
        rate = n_orig / raw.times[-1] * 60
        msg = f'Detected {rate_names[kind]} rate: {rate:5.1f} bpm'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))
        # Enough to start
        if len(proj_epochs) >= minimums[kind]:
            reject_ = _get_reject(
                subject=subject,
                session=session,
                reject=rejects[kind],
                ch_types=cfg.ch_types,
                param=f'ssp_reject_{kind}',
                epochs=proj_epochs,
            )
            proj_epochs.drop_bad(reject=reject_)
        # Still enough after rejection
        if len(proj_epochs) >= minimums[kind]:
            proj_epochs.apply_baseline((None, None))
            use = proj_epochs.average() if avg[kind] else proj_epochs
            fun = compute_proj_evoked if avg[kind] else compute_proj_epochs
            desc_prefix = (
                f'{kind.upper()}-'
                f'{proj_epochs.times[0]:0.3f}-'
                f'{proj_epochs.times[-1]:0.3f})'
            )
            projs[kind] = fun(
                use, meg=cfg.ssp_meg, **n_projs[kind], desc_prefix=desc_prefix)
            out_files[f'{kind}_epochs'] = out_files['proj'].copy().update(
                suffix=f'{kind}-epo', split=None, check=False)
            proj_epochs.save(out_files[f'{kind}_epochs'], overwrite=True)
        else:
            msg = (f'No {kind.upper()} projectors computed: got '
                   f'{len(proj_epochs)} good epochs < {minimums[kind]} '
                   f'(from {n_orig} original events).')
            logger.warning(**gen_log_kwargs(message=msg, subject=subject,
                                            session=session))
        del proj_epochs

    mne.write_proj(out_files['proj'], sum(projs.values(), []), overwrite=True)
    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        runs=get_runs(config=config, subject=subject),
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        eog_channels=config.eog_channels,
        deriv_root=get_deriv_root(config),
        ssp_reject_ecg=config.ssp_reject_ecg,
        ecg_proj_from_average=config.ecg_proj_from_average,
        ssp_reject_eog=config.ssp_reject_eog,
        eog_proj_from_average=config.eog_proj_from_average,
        min_ecg_epochs=config.min_ecg_epochs,
        min_eog_epochs=config.min_eog_epochs,
        n_proj_eog=config.n_proj_eog,
        n_proj_ecg=config.n_proj_ecg,
        ssp_meg=config.ssp_meg,
        ch_types=config.ch_types,
        decim=config.decim,
        use_maxwell_filter=config.use_maxwell_filter,
    )
    return cfg


@auto_script_path
def main(*, config) -> None:
    """Run SSP."""
    if config.spatial_filter != 'ssp':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config):
        parallel, run_func = parallel_func(run_ssp, config=config)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
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
