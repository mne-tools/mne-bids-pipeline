"""Noise covariance estimation.

Covariance matrices are computed and saved.
"""

import itertools
from typing import Optional
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_deriv_root,
    get_noise_cov_bids_path, _import_config,
)
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, save_logs, _sanitize_callable
from ..._parallel import get_parallel_backend, parallel_func


def get_input_fnames_cov(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    # short circuit to say: always re-run
    cov_type = _get_cov_type(cfg)
    in_files = dict()
    if cov_type == 'custom':
        in_files['__unknown_inputs__'] = 'custom noise_cov callable'
        return in_files
    if cov_type == 'raw':
        bids_path_raw_noise = BIDSPath(
            subject=subject,
            session=session,
            task=cfg.task,
            acquisition=cfg.acq,
            run=None,
            recording=cfg.rec,
            space=cfg.space,
            processing='filt',
            suffix='raw',
            extension='.fif',
            datatype=cfg.datatype,
            root=cfg.deriv_root,
            check=False
        )
        data_type = ('resting-state' if cfg.noise_cov == 'rest' else
                     'empty-room')
        if data_type == 'resting-state':
            bids_path_raw_noise.task = 'rest'
        else:
            bids_path_raw_noise.task = 'noise'
        in_files['raw'] = bids_path_raw_noise
    else:
        assert cov_type == 'epochs', cov_type
        processing = None
        if cfg.spatial_filter is not None:
            processing = 'clean'
        fname_epochs = BIDSPath(subject=subject,
                                session=session,
                                task=cfg.task,
                                acquisition=cfg.acq,
                                run=None,
                                recording=cfg.rec,
                                space=cfg.space,
                                extension='.fif',
                                suffix='epo',
                                processing=processing,
                                datatype=cfg.datatype,
                                root=cfg.deriv_root,
                                check=False)
        in_files['epochs'] = fname_epochs
    return in_files


def compute_cov_from_epochs(
        *, cfg, subject, session, tmin, tmax, in_files, out_files):
    epo_fname = in_files.pop('epochs')

    msg = "Computing regularized covariance based on epochs' baseline periods."
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f"Input:  {epo_fname.basename}"
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f"Output: {out_files['cov'].basename}"
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(epo_fname, preload=True)
    cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk',
                                 rank='info')
    return cov


def compute_cov_from_raw(*, cfg, subject, session, in_files, out_files):
    fname_raw = in_files.pop('raw')
    data_type = 'resting-state' if fname_raw.task == 'rest' else 'empty-room'
    msg = f'Computing regularized covariance based on {data_type} recording.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f'Input:  {fname_raw.basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f'Output: {out_files["cov"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    raw_noise = mne.io.read_raw_fif(fname_raw, preload=True)
    cov = mne.compute_raw_covariance(raw_noise, method='shrunk', rank='info')
    return cov


def retrieve_custom_cov(
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: None,
    out_files: dict,
):
    # This should be the only place we use config.noise_cov (rather than cfg.*
    # entries)
    config = _import_config()
    assert cfg.noise_cov == 'custom'
    assert callable(config.noise_cov)
    assert in_files == {}, in_files  # unknown

    # ... so we construct the input file we need here
    evoked_bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        processing=cfg.proc,
        recording=cfg.rec,
        space=cfg.space,
        suffix='ave',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )

    msg = ('Retrieving noise covariance matrix from custom user-supplied '
           'function')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f'Output: {out_files["cov"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    cov = config.noise_cov(evoked_bids_path)
    assert isinstance(cov, mne.Covariance)
    return cov


def _get_cov_type(cfg):
    if cfg.noise_cov == 'custom':
        return 'custom'
    elif cfg.noise_cov == 'rest':
        return 'raw'
    elif cfg.noise_cov == 'emptyroom' and 'eeg' not in cfg.ch_types:
        return 'raw'
    else:
        return 'epochs'


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_cov)
def run_covariance(*, cfg, subject, session, in_files):
    out_files = dict()
    out_files['cov'] = get_noise_cov_bids_path(
        cfg=cfg,
        subject=subject,
        session=session
    )
    cov_type = _get_cov_type(cfg)
    kwargs = dict(
        cfg=cfg, subject=subject, session=session,
        in_files=in_files, out_files=out_files)
    if cov_type == 'custom':
        cov = retrieve_custom_cov(**kwargs)
    elif cov_type == 'raw':
        cov = compute_cov_from_raw(**kwargs)
    else:
        tmin, tmax = cfg.noise_cov
        cov = compute_cov_from_epochs(tmin=tmin, tmax=tmax, **kwargs)
    cov.save(out_files['cov'], overwrite=True)
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
        space=config.space,
        proc=config.proc,
        spatial_filter=config.spatial_filter,
        ch_types=config.ch_types,
        deriv_root=get_deriv_root(config),
        run_source_estimation=config.run_source_estimation,
        noise_cov=_sanitize_callable(config.noise_cov),
    )
    return cfg


def main(*, config) -> None:
    """Run cov."""
    cfg = get_config(config=config)

    if not cfg.run_source_estimation:
        msg = 'Skipping, run_source_estimation is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    # Note that we're using config.noise_cov here and not adding it to
    # cfg, as in case it's a function, it won't work when running parallel jobs

    if config.noise_cov == "ad-hoc":
        msg = 'Skipping, using ad-hoc diagonal covariance …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config=config):
        parallel, run_func = parallel_func(run_covariance, config=config)
        logs = parallel(
            run_func(cfg=cfg, subject=subject, session=session)
            for subject, session in
            itertools.product(
                get_subjects(config),
                get_sessions(config),
            )
        )
    save_logs(config=config, logs=logs)
