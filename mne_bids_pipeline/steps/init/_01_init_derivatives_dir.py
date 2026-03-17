"""Initialize derivatives_dir.

Initialize the derivatives directory.
"""

from types import SimpleNamespace

from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

from mne_bids_pipeline._config_utils import _bids_kwargs, get_subjects_sessions
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._report import _open_report, _report_path
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run
from mne_bids_pipeline.typing import OutFilesT


@failsafe_run()
def init_dataset(cfg: SimpleNamespace, exec_params: SimpleNamespace) -> OutFilesT:
    """Prepare the pipeline directory in /derivatives."""
    out_files = dict()
    out_files["json"] = cfg.deriv_root / "dataset_description.json"
    logger.info(**gen_log_kwargs(message="Initializing output directories."))

    cfg.deriv_root.mkdir(exist_ok=True, parents=True)

    # Write a dataset_description.json for the pipeline
    ds_json = dict()
    ds_json["Name"] = cfg.PIPELINE_NAME + " outputs"
    ds_json["BIDSVersion"] = BIDS_VERSION
    ds_json["PipelineDescription"] = {
        "Name": cfg.PIPELINE_NAME,
        "Version": cfg.VERSION,
        "CodeURL": cfg.CODE_URL,
    }
    ds_json["SourceDatasets"] = {
        "URL": "n/a",
    }

    _write_json(out_files["json"], ds_json, overwrite=True)
    return _prep_out_files_path(exec_params=exec_params, out_files=out_files)


def init_subject_dirs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
) -> None:
    """Create processing data output directories for individual participants."""
    out_dir = cfg.deriv_root / f"sub-{subject}"
    if session is not None:
        out_dir /= f"ses-{session}"
    out_dir /= cfg.datatype

    out_dir.mkdir(exist_ok=True, parents=True)

    if not _report_path(cfg=cfg, subject=subject, session=session).fpath.is_file():
        with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session,
        ):
            pass


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        PIPELINE_NAME=config.PIPELINE_NAME,
        VERSION=config.VERSION,
        CODE_URL=config.CODE_URL,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Initialize the output directories."""
    init_dataset(cfg=get_config(config=config), exec_params=config.exec_params)
    # Don't bother with parallelization here as I/O operations are generally
    # not well parallelized (and this should be very fast anyway)
    for subject, sessions in get_subjects_sessions(config).items():
        for session in sessions:
            init_subject_dirs(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
