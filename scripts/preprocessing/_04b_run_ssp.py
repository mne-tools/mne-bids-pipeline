"""
===========
04. Run SSP
===========

Compute Signal Subspace Projections (SSP).
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne_bids import BIDSPath
from mne.utils import _pl

import config
from config import gen_log_kwargs, failsafe_run, _update_for_splits
from config import parallel_func, _script_path


logger = logging.getLogger('mne-bids-pipeline')


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


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_run_ssp)
def run_ssp(*, cfg, subject, session, in_files):
    # compute SSP on first run of raw
    raw_fnames = [in_files.pop(f'raw_run-{run}') for run in cfg.runs]

    # when saving proj, use run=None
    out_files = dict()
    out_files['proj'] = raw_fnames[0].copy().update(
        run=None, suffix='proj', split=None, processing=None, check=False)

    msg = (f'Input{_pl(raw_fnames)} ({len(raw_fnames)}): '
           f'{raw_fnames[0].basename}{_pl(raw_fnames, pl=" ...")}, ')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = (f'Output: {out_files["proj"].basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    raw = mne.concatenate_raws([
        mne.io.read_raw_fif(raw_fname_in) for raw_fname_in in raw_fnames])
    del raw_fnames
    msg = 'Computing SSPs for ECG'
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))

    ecg_projs = []
    ecg_epochs = create_ecg_epochs(raw)
    if cfg.ssp_meg == 'auto':
        cfg.ssp_meg = 'combined' if cfg.use_maxwell_filter else 'separate'
    if len(ecg_epochs) >= cfg.min_ecg_epochs:
        if cfg.ssp_reject_ecg == 'autoreject_global':
            reject_ecg_ = config.get_ssp_reject(
                ssp_type='ecg',
                epochs=ecg_epochs)
            ecg_projs, _ = compute_proj_ecg(raw,
                                            average=cfg.ecg_proj_from_average,
                                            reject=reject_ecg_,
                                            meg=cfg.ssp_meg,
                                            **cfg.n_proj_ecg)
        else:
            reject_ecg_ = config.get_ssp_reject(
                    ssp_type='ecg',
                    epochs=None)
            ecg_projs, _ = compute_proj_ecg(raw,
                                            average=cfg.ecg_proj_from_average,
                                            reject=reject_ecg_,
                                            meg=cfg.ssp_meg,
                                            **cfg.n_proj_ecg)

    if not ecg_projs:
        msg = ('Not enough ECG events could be found. No ECG projectors are '
               'computed.')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

    msg = 'Computing SSPs for EOG'
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))
    if cfg.eog_channels:
        ch_names = cfg.eog_channels
        assert all([ch_name in raw.ch_names for ch_name in ch_names])
    else:
        ch_names = None

    eog_projs = []
    eog_epochs = create_eog_epochs(raw)
    if len(eog_epochs) >= cfg.min_eog_epochs:
        if cfg.ssp_reject_eog == 'autoreject_global':
            reject_eog_ = config.get_ssp_reject(
                ssp_type='eog',
                epochs=eog_epochs)
            eog_projs, _ = compute_proj_eog(raw,
                                            average=cfg.eog_proj_from_average,
                                            reject=reject_eog_,
                                            **cfg.n_proj_eog)
        else:
            reject_eog_ = config.get_ssp_reject(
                    ssp_type='eog',
                    epochs=None)
            eog_projs, _ = compute_proj_eog(raw,
                                            average=cfg.eog_proj_from_average,
                                            reject=reject_eog_,
                                            **cfg.n_proj_eog)

    if not eog_projs:
        msg = ('Not enough EOG events could be found. No EOG projectors are '
               'computed.')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

    mne.write_proj(out_files['proj'], eog_projs + ecg_projs, overwrite=True)
    # TODO: Write the epochs as well for nice joint plots

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        runs=config.get_runs(subject=subject),
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        eog_channels=config.eog_channels,
        deriv_root=config.get_deriv_root(),
        ssp_reject_ecg=config.ssp_reject_ecg,
        ecg_proj_from_average=config.ecg_proj_from_average,
        ssp_reject_eog=config.ssp_reject_eog,
        eog_proj_from_average=config.eog_proj_from_average,
        n_proj_eog=config.n_proj_eog,
        n_proj_ecg=config.n_proj_ecg,
        ssp_meg=config.ssp_meg,
        use_maxwell_filter=config.use_maxwell_filter,
    )
    return cfg


def main():
    """Run SSP."""
    if config.spatial_filter != 'ssp':
        msg = 'Skipping …'
        with _script_path(__file__):
            logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_ssp)
        logs = parallel(
            run_func(
                cfg=get_config(subject=subject), subject=subject,
                session=session
            )
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
