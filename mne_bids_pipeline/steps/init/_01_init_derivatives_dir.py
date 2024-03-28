"""Initialize derivatives_dir.

Initialize the derivatives directory.
"""

from pathlib import Path
from types import SimpleNamespace

from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

from ..._config_utils import _bids_kwargs, get_sessions, get_subjects
from ..._logging import gen_log_kwargs, logger
from ..._run import _prep_out_files, failsafe_run


@failsafe_run()
def init_dataset(cfg: SimpleNamespace, exec_params: SimpleNamespace) -> dict[str, Path]:
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
    return _prep_out_files(
        exec_params=exec_params, out_files=out_files, bids_only=False
    )


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


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        PIPELINE_NAME=config.PIPELINE_NAME,
        VERSION=config.VERSION,
        CODE_URL=config.CODE_URL,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config):
    """Initialize the output directories."""
    init_dataset(cfg=get_config(config=config), exec_params=config.exec_params)
    # Don't bother with parallelization here as I/O operations are generally
    # not well parallelized (and this should be very fast anyway)
    for subject in get_subjects(config):
        for session in get_sessions(config):
            init_subject_dirs(
                cfg=get_config(
                    config=config,
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
