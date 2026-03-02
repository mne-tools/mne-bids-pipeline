"""Noise covariance estimation.

Covariance matrices are computed and saved.
"""

import contextlib
import logging
from collections.abc import Generator
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

from mne_bids_pipeline._config_import import _import_config
from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_rank,
    _get_ss,
    _restrict_analyze_channels,
    get_eeg_reference,
    get_noise_cov_bids_path,
)
from mne_bids_pipeline._io import _write_json
from mne_bids_pipeline._logging import _log_context, gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _all_conditions, _open_report, _sanitize_cond_tag
from mne_bids_pipeline._run import (
    _prep_out_files,
    _sanitize_callable,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_cov(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> InFilesT:
    cov_type = _get_cov_type(cfg)
    in_files = dict()
    fname_epochs = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        suffix="epo",
        processing="clean",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files["report_info"] = fname_epochs.copy().update(processing="clean")
    _update_for_splits(in_files, "report_info", single=True)
    fname_evoked = fname_epochs.copy().update(
        suffix="ave", processing=None, check=False
    )
    if fname_evoked.fpath.exists():
        in_files["evoked"] = fname_evoked
    if cov_type == "custom":
        in_files["__unknown_inputs__"] = "custom noise_cov callable"
        return in_files
    if cov_type == "raw":
        bids_path_raw_noise = BIDSPath(
            subject=subject,
            session=session,
            task=cfg.task,
            acquisition=cfg.acq,
            run=None,
            recording=cfg.rec,
            space=cfg.space,
            processing="clean",
            suffix="raw",
            extension=".fif",
            datatype=cfg.datatype,
            root=cfg.deriv_root,
            check=False,
        )
        if cfg.noise_cov == "rest":
            bids_path_raw_noise.task = "rest"
        else:
            bids_path_raw_noise.task = "noise"
        in_files["raw"] = bids_path_raw_noise
    else:
        assert cov_type == "epochs", cov_type
        in_files["epochs"] = fname_epochs
        _update_for_splits(in_files, "epochs", single=True)
    return in_files


def compute_cov_rank_from_epochs(
    *,
    tmin: float | None,
    tmax: float | None,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
    out_files: InFilesT,
) -> tuple[mne.Covariance, dict[str, int]]:
    epo_fname = in_files.pop("epochs")

    msg = "Computing regularized covariance based on epochs' baseline periods."
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Input:  {epo_fname.basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['cov'].basename}"
    logger.info(**gen_log_kwargs(message=msg))

    epochs = mne.read_epochs(epo_fname, preload=True)
    rank = _get_rank(cfg=cfg, subject=subject, session=session, inst=epochs)
    cov = mne.compute_covariance(
        epochs,
        tmin=tmin,
        tmax=tmax,
        method=cfg.noise_cov_method,
        rank=rank,
        verbose="error",  # TODO: not baseline corrected, maybe problematic?
    )
    return cov, rank


def compute_cov_rank_from_raw(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
    out_files: InFilesT,
) -> tuple[mne.Covariance, dict[str, int]]:
    fname_raw = in_files.pop("raw")
    run_msg = "resting-state" if fname_raw.task == "rest" else "empty-room"
    msg = f"Computing regularized covariance based on {run_msg} recording."
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Input:  {fname_raw.basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['cov'].basename}"
    logger.info(**gen_log_kwargs(message=msg))

    raw_noise = mne.io.read_raw_fif(fname_raw, preload=True)
    rank = _get_rank(cfg=cfg, subject=subject, session=session, inst=raw_noise)
    cov = mne.compute_raw_covariance(
        raw_noise,
        method=cfg.noise_cov_method,
        rank=rank,
    )
    return cov, rank


def retrieve_custom_cov_rank(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
    out_files: InFilesT,
) -> tuple[mne.Covariance, dict[str, int]]:
    # This should be the only place we use config.noise_cov (rather than cfg.*
    # entries)
    with _log_context(logging.CRITICAL):
        config = _import_config(
            config_path=exec_params.config_path,
            check=False,
        )
    assert cfg.noise_cov == "custom"
    assert callable(config.noise_cov)
    assert in_files == {}, in_files  # unknown

    # ... so we construct the input file we need here
    epochs_bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        processing="clean",
        recording=cfg.rec,
        space=cfg.space,
        suffix="epo",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    info_bids_path = epochs_bids_path
    if not info_bids_path.fpath.exists():
        info_bids_path = info_bids_path.copy().update(split="01")

    msg = "Retrieving noise covariance matrix from custom user-supplied function"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['cov'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    info = mne.io.read_info(info_bids_path)

    cov = config.noise_cov(epochs_bids_path)
    assert isinstance(cov, mne.Covariance)
    rank = _get_rank(
        cfg=cfg,
        subject=subject,
        session=session,
        inst=cov,
        info=info,
    )
    return cov, rank


def _get_cov_type(cfg: SimpleNamespace) -> str:
    if cfg.noise_cov == "custom":
        return "custom"
    elif cfg.noise_cov == "rest":
        return "raw"
    elif cfg.noise_cov == "emptyroom" and "eeg" not in cfg.ch_types:
        return "raw"
    else:
        return "epochs"


# Workaround for https://github.com/mne-tools/mne-python/pull/13595
# to get MNE < 1.11.1 to tolerate rank=dict(meg=...)
@contextlib.contextmanager
def _fake_sss_context() -> Generator[None, None, None]:
    """Fake SSS context manager for MNE < 1.11.1."""
    import mne.viz.utils

    orig = mne.viz.utils._check_sss

    def replacement(*args, **kwargs):
        out = list(orig(*args, **kwargs))
        assert len(out) == 3 and isinstance(out[-1], bool)
        out[-1] = out[-2]  # has_sss = has_meg (which really means: mag + grad)
        return tuple(out)

    mne.viz.utils._check_sss = replacement  # type: ignore[assignment]
    try:
        yield
    finally:
        mne.viz.utils._check_sss = orig


@failsafe_run(
    get_input_fnames=get_input_fnames_cov,
)
def run_covariance(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None = None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    out_files = dict()
    out_files["cov"] = get_noise_cov_bids_path(
        cfg=cfg, subject=subject, session=session
    )
    out_files["rank"] = out_files["cov"].copy().update(suffix="rank", extension=".json")
    cov_type = _get_cov_type(cfg)
    fname_info = in_files.pop("report_info")
    fname_evoked = in_files.pop("evoked", None)
    if cov_type == "custom":
        cov, rank = retrieve_custom_cov_rank(
            cfg=cfg,
            subject=subject,
            session=session,
            in_files=in_files,
            out_files=out_files,
            exec_params=exec_params,
        )
    elif cov_type == "raw":
        cov, rank = compute_cov_rank_from_raw(
            cfg=cfg,
            subject=subject,
            session=session,
            in_files=in_files,
            out_files=out_files,
            exec_params=exec_params,
        )
    else:
        tmin, tmax = cfg.noise_cov
        cov, rank = compute_cov_rank_from_epochs(
            tmin=tmin,
            tmax=tmax,
            cfg=cfg,
            subject=subject,
            session=session,
            in_files=in_files,
            out_files=out_files,
            exec_params=exec_params,
        )
    cov.save(out_files["cov"], overwrite=True)
    _write_json(out_files["rank"], rank)

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        msg = "Rendering noise covariance matrix and corresponding SVD."
        logger.info(**gen_log_kwargs(message=msg))
        section = "Noise covariance"
        report.add_html(
            html=f"<code>{rank}</code>",
            title="Rank",
            section=section,
            tags=("covariance",),
            replace=True,
        )
        report.add_covariance(
            cov=cov,
            info=fname_info,
            title=section,
            replace=True,
        )
        if fname_evoked is not None:
            msg = "Rendering whitened evoked data."
            logger.info(**gen_log_kwargs(message=msg))
            all_evoked = mne.read_evokeds(fname_evoked)
            assert isinstance(all_evoked, list)
            conditions = _all_conditions(cfg=cfg)
            assert len(all_evoked) == len(conditions)
            for evoked, condition in zip(all_evoked, conditions):
                _restrict_analyze_channels(evoked, cfg)
                tags: tuple[str, ...] = (
                    "evoked",
                    "covariance",
                    _sanitize_cond_tag(condition),
                )
                title = f"Whitening: {condition}"
                if condition not in cfg.conditions:
                    tags = tags + ("contrast",)
                with _fake_sss_context():
                    fig = evoked.plot_white(cov, rank=rank, verbose="error")
                report.add_figure(
                    fig=fig,
                    title=title,
                    tags=tags,
                    section=section,
                    replace=True,
                )
                plt.close(fig)

    assert len(in_files) == 0, in_files
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        ch_types=config.ch_types,
        run_source_estimation=config.run_source_estimation,
        noise_cov=_sanitize_callable(config.noise_cov),
        conditions=config.conditions,
        contrasts=config.contrasts,
        analyze_channels=config.analyze_channels,
        eeg_reference=get_eeg_reference(config),
        noise_cov_method=config.noise_cov_method,
        cov_rank=config.cov_rank,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run cov."""
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    # Note that we're using config.noise_cov here and not adding it to
    # cfg, as in case it's a function, it won't work when running parallel jobs

    if config.noise_cov == "ad-hoc":
        msg = "Skipping, using ad-hoc diagonal covariance …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    ss = _get_ss(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_covariance, exec_params=config.exec_params, n_iter=len(ss)
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject, session in ss
        )
    save_logs(config=config, logs=logs)
