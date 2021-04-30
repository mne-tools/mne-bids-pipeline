import logging
import itertools
from typing import Optional

from mne.parallel import parallel_func
from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


def init_dataset() -> None:
    """Prepare the pipeline directory in /derivatives.
    """
    msg = "Initializing output directories."
    logger.info(gen_log_message(step=1, message=msg))

    deriv_root = config.get_deriv_root()
    deriv_root.mkdir(exist_ok=True, parents=True)

    # Write a dataset_description.json for the pipeline
    ds_json = dict()
    ds_json['Name'] = config.PIPELINE_NAME + ' outputs'
    ds_json['BIDSVersion'] = BIDS_VERSION
    ds_json['PipelineDescription'] = {
        'Name': config.PIPELINE_NAME,
        'Version': config.VERSION,
        'CodeURL': config.CODE_URL,
    }
    ds_json['SourceDatasets'] = {
        'URL': 'n/a',
    }

    fname = deriv_root / 'dataset_description.json'
    _write_json(fname, ds_json, overwrite=True)


def init_subject_dirs(
    *,
    subject: str,
    session: Optional[str] = None
) -> None:
    """Create processing data output directories for individual participants.
    """
    deriv_root = config.get_deriv_root()
    datatype = config.get_datatype()

    out_dir = deriv_root / f'sub-{subject}'
    if session is not None:
        out_dir /= f'ses-{session}'
    out_dir /= datatype

    out_dir.mkdir(exist_ok=True, parents=True)


@failsafe_run(on_error=on_error)
def main():
    """Initialize the output directories."""
    msg = 'Running: Initializing output directories.'
    logger.info(gen_log_message(step=1, message=msg))

    init_dataset()
    parallel, run_func, _ = parallel_func(init_subject_dirs,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject=subject, session=session)
             for subject, session in
             itertools.product(config.get_subjects(),
                               config.get_sessions()))

    msg = 'Completed: Initializing output directories.'
    logger.info(gen_log_message(step=1, message=msg))


if __name__ == '__main__':
    main()
