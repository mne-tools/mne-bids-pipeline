"""Compute SSP.

Signal subspace projections (SSP) vectors are computed from EOG and ECG signals.
These are often also referred to as PCA vectors.
"""

from types import SimpleNamespace

import mne
from mne import compute_proj_epochs, compute_proj_evoked
from mne.preprocessing import find_ecg_events, find_eog_events
from mne_bids import BIDSPath

from mne_bids_pipeline._config_import import ConfigError
from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_ss,
    _pl,
    _proj_path,
    get_ecg_channel,
    get_eog_channels,
    get_runs_tasks,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._reject import _get_reject
from mne_bids_pipeline._report import _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, IntArrayT, OutFilesT


def _find_ecg_events(raw: mne.io.Raw, ch_name: str | None) -> IntArrayT:
    """Wrap find_ecg_events to use the same defaults as create_ecg_events."""
    out: IntArrayT = find_ecg_events(raw, ch_name=ch_name, l_freq=8, h_freq=16)[0]
    return out


def get_input_fnames_run_ssp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> InFilesT:
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        extension=".fif",
        check=False,
    )
    in_files = dict()
    for run, task in cfg.runs_tasks:
        key = f"raw_task-{task}_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, task=task, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_ssp,
)
def run_ssp(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    # compute SSP on all runs of raw
    raw_fnames = [
        in_files.pop(f"raw_task-{task}_run-{run}") for run, task in cfg.runs_tasks
    ]

    out_files = dict(proj=_proj_path(cfg=cfg, subject=subject, session=session))
    msg = (
        f"Input{_pl(raw_fnames)} ({len(raw_fnames)}): "
        f"{raw_fnames[0].basename}{_pl(raw_fnames, pl=' ...')}"
    )
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['proj'].basename}"
    logger.info(**gen_log_kwargs(message=msg))

    raw = mne.concatenate_raws(
        [mne.io.read_raw_fif(raw_fname_in) for raw_fname_in in raw_fnames]
    )
    del raw_fnames

    projs: dict[str, list[mne.Projection]] = dict()
    proj_kinds = ("ecg", "eog")
    rate_names = dict(ecg="heart", eog="blink")
    minimums = dict(ecg=cfg.min_ecg_epochs, eog=cfg.min_eog_epochs)
    rejects = dict(ecg=cfg.ssp_reject_ecg, eog=cfg.ssp_reject_eog)
    avg = dict(ecg=cfg.ecg_proj_from_average, eog=cfg.eog_proj_from_average)
    n_projs = dict(ecg=cfg.n_proj_ecg, eog=cfg.n_proj_eog)

    eog_chs_subj_sess = get_eog_channels(cfg.eog_channels, subject, session)

    ch_name_ecg: str | None = None
    ch_name_eog: str | list[str] | None = None
    if eog_chs_subj_sess:
        ch_name_eog = list(eog_chs_subj_sess)
        assert ch_name_eog is not None
        assert all(ch_name in raw.ch_names for ch_name in ch_name_eog)
    if cfg.ssp_ecg_channel:
        ch_name_ecg = get_ecg_channel(
            ecg_channel=cfg.ssp_ecg_channel, subject=subject, session=session
        )
        if ch_name_ecg not in raw.ch_names:
            raise ConfigError(
                f"SSP ECG channel '{ch_name_ecg}' not found in data for "
                f"subject {subject}, session {session}"
            )
    if cfg.ssp_meg == "auto":
        cfg.ssp_meg = "combined" if cfg.use_maxwell_filter else "separate"
    for kind in proj_kinds:
        projs[kind] = []
        if not any(n_projs[kind].values()):
            continue
        if kind == "ecg":
            assert isinstance(ch_name_ecg, str | None)
            events = _find_ecg_events(raw=raw, ch_name=ch_name_ecg)
        else:
            events = find_eog_events(raw=raw, ch_name=ch_name_eog)
        n_orig = len(events)
        rate = n_orig / raw.times[-1] * 60
        bpm_msg = f"{rate:5.1f} bpm"
        msg = f"Detected {rate_names[kind]} rate: {bpm_msg}"
        logger.info(**gen_log_kwargs(message=msg))
        # Enough to create epochs
        if len(events) < minimums[kind]:
            msg = (
                f"No {kind.upper()} projectors computed: got "
                f"{len(events)} original events < {minimums[kind]} {bpm_msg}"
            )
            logger.warning(**gen_log_kwargs(message=msg))
            continue
        out_files[f"events_{kind}"] = (
            out_files["proj"]
            .copy()
            .update(suffix=f"{kind}-eve", split=None, check=False, extension=".txt")
        )
        mne.write_events(out_files[f"events_{kind}"], events, overwrite=True)
        proj_epochs = mne.Epochs(
            raw,
            events=events,
            event_id=events[0, 2],
            tmin=-0.5,
            tmax=0.5,
            proj=False,
            baseline=(None, None),
            reject_by_annotation=True,
            preload=True,
            decim=cfg.epochs_decim,
        )
        if len(proj_epochs) >= minimums[kind]:
            reject_ = _get_reject(
                subject=subject,
                session=session,
                reject=rejects[kind],
                ch_types=cfg.ch_types,
                param=f"ssp_reject_{kind}",
                epochs=proj_epochs,
            )
            proj_epochs.drop_bad(reject=reject_)
        # Still enough after rejection
        if len(proj_epochs) >= minimums[kind]:
            use = proj_epochs.average() if avg[kind] else proj_epochs
            fun = compute_proj_evoked if avg[kind] else compute_proj_epochs
            desc_prefix = (
                f"{kind.upper()}-"
                f"{proj_epochs.times[0]:0.3f}-"
                f"{proj_epochs.times[-1]:0.3f})"
            )
            projs[kind] = fun(
                use, meg=cfg.ssp_meg, **n_projs[kind], desc_prefix=desc_prefix
            )
            out_files[f"epochs_{kind}"] = (
                out_files["proj"]
                .copy()
                .update(suffix=f"{kind}-epo", split=None, check=False)
            )
            proj_epochs.save(out_files[f"epochs_{kind}"], overwrite=True)
        else:
            msg = (
                f"No {kind.upper()} projectors computed: got "
                f"{len(proj_epochs)} good epochs < {minimums[kind]} "
                f"(from {n_orig} original events; {bpm_msg})."
            )
            logger.warning(**gen_log_kwargs(message=msg))
        del proj_epochs

    mne.write_proj(out_files["proj"], sum(projs.values(), []), overwrite=True)
    assert len(in_files) == 0, in_files.keys()
    del projs

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        for kind in proj_kinds:
            key = f"epochs_{kind}"
            if key not in out_files:
                continue

            msg = f"Adding {kind.upper()} SSP to report."
            logger.info(**gen_log_kwargs(message=msg))
            proj_epochs = mne.read_epochs(out_files[f"epochs_{kind}"])
            these_projs: list[mne.Projection] = mne.read_proj(out_files["proj"])
            these_projs = [p for p in these_projs if kind.upper() in p["desc"]]
            assert len(these_projs), len(these_projs)  # should exist if the epochs do
            picks_trace: str | list[str] | None = None
            if kind == "ecg":
                if cfg.ssp_ecg_channel:
                    picks_trace = [
                        get_ecg_channel(
                            ecg_channel=cfg.ssp_ecg_channel,
                            subject=subject,
                            session=session,
                        )
                    ]
                elif "ecg" in proj_epochs:
                    picks_trace = "ecg"
            else:
                assert kind == "eog"
                if eog_chs_subj_sess:
                    # convert to list for compatibility of type annotations
                    picks_trace = list(eog_chs_subj_sess)
                elif "eog" in proj_epochs:
                    picks_trace = "eog"
            fig = mne.viz.plot_projs_joint(
                these_projs, proj_epochs.average(picks="all"), picks_trace=picks_trace
            )
            assert isinstance(proj_epochs.drop_log, tuple)
            caption = (
                f"Computed using {len(proj_epochs)} epochs "
                f"(from {len(proj_epochs.drop_log)} original events)"
            )
            report.add_figure(
                fig,
                title=f"SSP: {kind.upper()}",
                caption=caption,
                tags=("ssp", kind),
                replace=True,
            )
            plt.close(fig)
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        eog_channels=config.eog_channels,
        ssp_ecg_channel=config.ssp_ecg_channel,
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
        epochs_decim=config.epochs_decim,
        use_maxwell_filter=config.use_maxwell_filter,
        runs_tasks=get_runs_tasks(
            config=config, subject=subject, session=session, which=("runs", "rest")
        ),
        processing="filt" if config.regress_artifact is None else "regress",
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run SSP."""
    if config.spatial_filter != "ssp":
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ss = _get_ss(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_ssp, exec_params=config.exec_params, n_iter=len(ss)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject, session in ss
        )
    save_logs(config=config, logs=logs)
