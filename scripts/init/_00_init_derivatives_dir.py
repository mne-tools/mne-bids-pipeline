"""Initialize derivatives_dir.

Initialize the derivatives directory.
"""

import logging
import itertools
from typing import Optional
from types import SimpleNamespace

from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

import config
from config import gen_log_kwargs, failsafe_run
from config import parallel_func

logger = logging.getLogger('mne-bids-pipeline')


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


def init_subject_dirs(
    *,
    cfg,
    subject: str,
    session: Optional[str] = None
) -> None:
    """Create processing data output directories for individual participants.
    """
    out_dir = cfg.deriv_root / f'sub-{subject}'
    if session is not None:
        out_dir /= f'ses-{session}'
    out_dir /= cfg.datatype

    out_dir.mkdir(exist_ok=True, parents=True)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        datatype=config.get_datatype(),
        deriv_root=config.get_deriv_root(),
        PIPELINE_NAME=config.PIPELINE_NAME,
        VERSION=config.VERSION,
        CODE_URL=config.CODE_URL
    )
    return cfg


@failsafe_run(script_path=__file__)
def main():
    """Initialize the output directories."""
    with config.get_parallel_backend():
        init_dataset(cfg=get_config())
        parallel, run_func = parallel_func(init_subject_dirs)
        parallel(run_func(cfg=get_config(), subject=subject, session=session)
                 for subject, session in
                 itertools.product(
                    config.get_subjects(),
                    config.get_sessions()
                )
            )


if __name__ == '__main__':
    main()
