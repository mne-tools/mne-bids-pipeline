import mne
import numpy as np
from mne.preprocessing import oversampled_temporal_projection
from mne_bids import BIDSPath, read_raw_bids
from mne_bids_pipeline._config_utils import _get_ssrt
from mne_bids_pipeline._run import _prep_out_files
from mne_bids_pipeline._import_data import (
    _bads_path,
    _get_mf_reference_path,
    _get_run_rest_noise_path,
    _import_data_kwargs,
    _read_raw_msg,
    import_er_data,
    import_experimental_data,
)
from mne_bids_pipeline._01a_data_quality import get_input_fname_data_quality


def get_input_fnames_otp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by filter_data function."""
    return _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
        add_bads=(kind == "orig"),
    )


@failsafe_run(
    get_input_fnames=get_input_fnames_data_quality,
    sidecars=True,
)
def apply_otp(
        *,
        cfg: SimpleNamespace,
        exec_params: SimpleNamespace,
        subject: str,
        session: str | None,
        run: str | None,
        task: str | None,
        in_files: InFilesT,
)-> OutFilesT:
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads")
    msg, run_type = _read_raw_msg(bids_path_in=bids_path_in, run=run, task=task)
    logger.info(**gen_log_kwargs(message=msg))

    if run is None and task == "noise":
        raw = import_er_data(
            cfg=cfg,
            exec_params=exec_params,
            bids_path_er_in=bids_path_in,
            bids_path_er_bads_in=None,
            bids_path_ref_in=bids_path_ref_in,
            bids_path_ref_bads_in=None,
            prepare_maxwell_filter=False,
        )
    else:
        data_is_rest = run is None and task == "rest"
        raw = import_experimental_data(
            cfg=cfg,
            exec_params=exec_params,
            bids_path_in=bids_path_in,
            bids_path_bads_in=None,
            data_is_rest=data_is_rest,
        )

    out_files = dict()
    out_files[in_key] = bids_path_in.copy().update(
        root=cfg.deriv_root,
        subject=subject,  # save under subject's directory so all files are there
        session=session,
        processing="otp",
        extension=".fif",
        suffix="raw",
        split=None,
        task=task,
        run=run,
        check=False,
    )

    raw.load_data()
    raw = oversampled_temporal_projection(raw, duration=cfg.otp_duration, picks=None,)

    out_files[in_key].fpath.parent.mkdir(exist_ok=True, parents=True)

    raw.save(
        out_files[in_key],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._raw_split_size,
    )

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        msg = "Denoising raw with oversampled temporal projection..."
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files[in_key],
            title_prefix="Raw (OTP)",
            tags=("otp",),
            raw=raw,
        )

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> SimpleNamespace:
    # picks, duration
    cfg = SinmpleNamespace(
        duration=config.otp_duration,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run assess_data_quality."""
    ssrt = _get_ssrt(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            apply_otp,
            exec_params=config.exec_params,
            n_iter=len(ssrt),
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject, session=session),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )

    save_logs(config=config, logs=logs)

