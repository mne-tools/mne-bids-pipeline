"""Initialize derivatives_dir.

Initialize the derivatives directory.
"""

from typing import Optional
from types import SimpleNamespace

from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

from ..._config_utils import get_datatype, get_subjects, get_sessions
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run


def init_dataset(cfg) -> None:
    """Prepare the pipeline directory in /derivatives.
    """
    fname_json = cfg.deriv_root / 'dataset_description.json'
    if fname_json.is_file():
        return  # already exists
    msg = "Initializing output directories."
    logger.info(**gen_log_kwargs(message=msg))

    cfg.deriv_root.mkdir(exist_ok=True, parents=True)

    # Write a dataset_description.json for the pipeline
    ds_json = dict()
    ds_json['Name'] = cfg.PIPELINE_NAME + ' outputs'
    ds_json['BIDSVersion'] = BIDS_VERSION
    ds_json['PipelineDescription'] = {
        'Name': cfg.PIPELINE_NAME,
        'Version': cfg.VERSION,
        'CodeURL': cfg.CODE_URL,
    }
    ds_json['SourceDatasets'] = {
        'URL': 'n/a',
    }

    _write_json(fname_json, ds_json, overwrite=True)


@failsafe_run()
def init_subject_dirs(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> None:
    """Create processing data output directories for individual participants.
    """
    out_dir = cfg.deriv_root / f'sub-{subject}'
    if session is not None:
        out_dir /= f'ses-{session}'
    out_dir /= cfg.datatype

    out_dir.mkdir(exist_ok=True, parents=True)


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        datatype=get_datatype(config),
        deriv_root=config.deriv_root,
        PIPELINE_NAME=config.PIPELINE_NAME,
        VERSION=config.VERSION,
        CODE_URL=config.CODE_URL,
    )
    return cfg


def main(*, config):
    """Initialize the output directories."""
    init_dataset(cfg=get_config(config=config))
    # Don't bother with parallelization here as I/O operations are generally
    # not well paralellized (and this should be very fast anyway)
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
