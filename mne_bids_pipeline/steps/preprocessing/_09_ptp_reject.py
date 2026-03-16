"""Remove epochs based on PTP amplitudes.

Epochs containing peak-to-peak (PTP) above the thresholds defined
in the 'reject' parameter are removed from the data.

This step will drop epochs containing non-biological artifacts
but also epochs containing biological artifacts not sufficiently
corrected by the ICA or the SSP processing.
"""

from types import SimpleNamespace

import autoreject
import mne
import numpy as np
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import _bids_kwargs, _get_sst
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._reject import _get_reject
from mne_bids_pipeline._report import _get_prefix_tags, _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT

from ._07_make_epochs import _add_epochs_image_kwargs


def get_input_fnames_drop_ptp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> InFilesT:
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix="epo",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files = dict()
    in_files["epochs"] = bids_path.copy().update(processing=cfg.spatial_filter)
    _update_for_splits(in_files, "epochs", single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_drop_ptp,
)
def drop_ptp(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    import matplotlib.pyplot as plt

    out_files = dict()
    out_files["epochs"] = (
        in_files["epochs"]
        .copy()
        .update(
            processing="clean",
            task=task,
            split=None,
        )
    )
    msg = f"Input:  {in_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))
    msg = f"Output: {out_files['epochs'].basename}"
    logger.info(**gen_log_kwargs(message=msg))

    # Get rejection parameters and drop bad epochs
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    if cfg.reject == "autoreject_local":
        msg = (
            "Using autoreject to find and repair bad epochs (interpolating bad "
            "segments)"
        )
        logger.info(**gen_log_kwargs(message=msg))

        ar = autoreject.AutoReject(
            n_interpolate=np.array(cfg.autoreject_n_interpolate),
            random_state=cfg.random_state,
            n_jobs=exec_params.n_jobs,
            verbose=False,
        )
        n_epochs_before_reject = len(epochs)
        epochs, reject_log = ar.fit_transform(epochs, return_log=True)
        n_epochs_after_reject = len(epochs)
        assert (
            n_epochs_before_reject - n_epochs_after_reject
            == reject_log.bad_epochs.sum()
        )

        msg = (
            f"autoreject marked {reject_log.bad_epochs.sum()} epochs as bad "
            f"(cross-validated n_interpolate limit: {ar.n_interpolate_})"
        )
        logger.info(**gen_log_kwargs(message=msg))
    else:
        reject = _get_reject(
            subject=subject,
            session=session,
            reject=cfg.reject,
            ch_types=cfg.ch_types,
            param="reject",
            epochs=epochs,
        )

        if cfg.spatial_filter == "ica" and cfg.ica_reject != "autoreject_local":
            ica_reject = _get_reject(
                subject=subject,
                session=session,
                reject=cfg.ica_reject,
                ch_types=cfg.ch_types,
                param="ica_reject",
            )
        else:
            ica_reject = None

        if ica_reject is not None:
            for ch_type, threshold in ica_reject.items():
                if ch_type in reject and threshold < reject[ch_type]:
                    # This can only ever happen in case of
                    # reject = 'autoreject_global'
                    msg = (
                        f"Adjusting PTP rejection threshold proposed by "
                        f"autoreject, as it is greater than ica_reject: "
                        f"{ch_type}: {reject[ch_type]} -> {threshold}"
                    )
                    logger.info(**gen_log_kwargs(message=msg))
                    reject[ch_type] = threshold

        n_epochs_before_reject = len(epochs)
        epochs.reject_tmin = cfg.reject_tmin
        epochs.reject_tmax = cfg.reject_tmax
        epochs.drop_bad(reject=reject)
        n_epochs_after_reject = len(epochs)
        n_epochs_rejected = n_epochs_before_reject - n_epochs_after_reject

        msg = (
            f"Removed {n_epochs_rejected} of {n_epochs_before_reject} epochs via PTP "
            f"rejection thresholds: {reject}"
        )
        logger.info(**gen_log_kwargs(message=msg))

    if 0 < n_epochs_after_reject < 0.5 * n_epochs_before_reject:
        msg = (
            "More than 50% of all epochs rejected. Please check the "
            "rejection thresholds."
        )
        logger.warning(**gen_log_kwargs(message=msg))
    elif n_epochs_after_reject == 0:
        rejection_type = (
            cfg.reject
            if cfg.reject in ["autoreject_global", "autoreject_local"]
            else "PTP-based"
        )
        raise RuntimeError(
            f"No epochs remaining after {rejection_type} rejection. Cannot continue."
        )

    msg = f"Saving {n_epochs_after_reject} cleaned, baseline-corrected epochs …"

    epochs.apply_baseline(cfg.baseline)
    epochs.save(
        out_files["epochs"],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._epochs_split_size,
    )
    _update_for_splits(out_files, "epochs")
    assert len(in_files) == 0, in_files.keys()

    # Report
    msg = "Adding cleaned epochs to report."
    logger.info(**gen_log_kwargs(message=msg))
    # Add PSD plots for 30s of data or all epochs if we have less available
    psd = True if len(epochs) * (epochs.tmax - epochs.tmin) < 30 else 30.0
    prefix, extra_tags = _get_prefix_tags(cfg=cfg, task=task)
    tags = ("epochs", "clean") + extra_tags
    kind = cfg.reject if isinstance(cfg.reject, str) else "Rejection"
    title = f"Epochs (clean){prefix}"
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session, task=task
    ) as report:
        if cfg.reject == "autoreject_local":
            caption = (
                f"Autoreject was run to produce cleaner epochs. "
                f"{reject_log.bad_epochs.sum()} epochs were rejected because more than "
                f"{ar.n_interpolate_} channels were bad (cross-validated n_interpolate "
                f"limit; excluding globally bad and non-data channels, shown in white)."
            )
            fig = reject_log.plot(orientation="horizontal", aspect="auto", show=False)
            report.add_figure(
                fig=fig,
                title=f"{kind} cleaning",
                caption=caption,
                section=title,
                tags=tags,
                replace=True,
            )
            plt.close(fig)
            del caption
        else:
            report.add_html(
                html=f"<code>{reject}</code>",
                title=f"{kind} thresholds",
                section=title,
                replace=True,
                tags=tags,
            )

        report.add_epochs(
            epochs=epochs,
            title=title,
            psd=psd,
            drop_log_ignore=(),
            tags=tags,
            replace=True,
            **_add_epochs_image_kwargs(cfg=cfg),
        )
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        baseline=config.baseline,
        reject_tmin=config.reject_tmin,
        reject_tmax=config.reject_tmax,
        spatial_filter=config.spatial_filter,
        ica_reject=config.ica_reject,
        reject=config.reject,
        autoreject_n_interpolate=config.autoreject_n_interpolate,
        random_state=config.random_state,
        ch_types=config.ch_types,
        _epochs_split_size=config._epochs_split_size,
        report_add_epochs_image_kwargs=config.report_add_epochs_image_kwargs,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run epochs."""
    sst = _get_sst(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            drop_ptp, exec_params=config.exec_params, n_iter=len(sst)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                task=task,
            )
            for subject, session, task in sst
        )
    save_logs(config=config, logs=logs)
