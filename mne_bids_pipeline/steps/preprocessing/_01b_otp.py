import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids


def get_input_fnames_otp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by _apply_otp function."""
    in_files: InFilesT = _get_run_rest_noise_path(
        run=run,
        task=task,
        kind="orig",
        mf_reference_run=cfg.mf_reference_run,
        mf_reference_task=cfg.mf_reference_task,
        cfg=cfg,
        subject=subject,
        session=session,
    )
    # When doing autobad for the noise run, we also need the reference run
    if _do_mf_autobad(cfg=cfg) and run is None and task == "noise":
        in_files.update(
            _get_mf_reference_path(
                cfg=cfg,
                subject=subject,
                session=session,
            )
        )

    # set calibration and crosstalk files (if provided)
    if _do_mf_autobad(cfg=cfg):
        # add these explicitly to in_files (duplicating with cfg) for proper caching
        if cfg.mf_cal_fname is not None:
            in_files["mf_cal_fname"] = cfg.mf_cal_fname
        if cfg.mf_ctc_fname is not None:
            in_files["mf_ctc_fname"] = cfg.mf_ctc_fname

    return in_files
