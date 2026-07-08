"""Initialize derivatives_dir.

Initialize the derivatives directory.
"""

from types import SimpleNamespace

from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

from mne_bids_pipeline._config_utils import _bids_kwargs, get_subjects_sessions
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._report import _open_report, _report_path
from mne_bids_pipeline._run import (
    _prep_out_files,
    _prep_out_files_path,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_init_dataset(*, cfg: SimpleNamespace) -> InFilesT:
    """Get input filenames for init_dataset."""
    return dict()


@failsafe_run(get_input_fnames=get_input_fnames_init_dataset)
def init_dataset(
    cfg: SimpleNamespace, exec_params: SimpleNamespace, in_files: InFilesT
) -> OutFilesT:
    """Prepare the pipeline directory in /derivatives."""
    assert not in_files, "init_dataset should not receive any input files"
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


def get_input_fnames_init_subject_dirs(
    *, cfg: SimpleNamespace, subject: str, session: str | None
) -> InFilesT:
    """Get input filenames for init_subject_dirs."""
    return dict()


@failsafe_run(get_input_fnames=get_input_fnames_init_subject_dirs)
def init_subject_dirs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    """Create processing data output directories for individual participants."""
    assert not in_files, "init_subject_dirs should not receive any input files"
    out_dir = cfg.deriv_root / f"sub-{subject}"
    if session is not None:
        out_dir /= f"ses-{session}"
    out_dir /= cfg.datatype

    out_dir.mkdir(exist_ok=True, parents=True)

    out_files = dict()
    out_files["report"] = _report_path(cfg=cfg, subject=subject, session=session)
    if not out_files["report"].fpath.is_file():
        with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session,
        ):
            pass
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config_init_dataset(
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


def get_config_init_subject_dirs(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    # Deliberately excludes PIPELINE_NAME/VERSION/CODE_URL (unlike
    # get_config_init_dataset): those aren't needed here, and including them would
    # cause unnecessary cache misses whenever they change (e.g. between releases).
    return SimpleNamespace(**_bids_kwargs(config=config))


def main(*, config: SimpleNamespace) -> None:
    """Initialize the output directories."""
    logs = [
        init_dataset(
            cfg=get_config_init_dataset(config=config), exec_params=config.exec_params
        )
    ]
    # Don't bother with parallelization here as I/O operations are generally
    # not well parallelized (and this should be very fast anyway)
    for subject, sessions in get_subjects_sessions(config).items():
        for session in sessions:
            logs.append(
                init_subject_dirs(
                    cfg=get_config_init_subject_dirs(config=config),
                    exec_params=config.exec_params,
                    subject=subject,
                    session=session,
                )
            )
    save_logs(config=config, logs=logs)
