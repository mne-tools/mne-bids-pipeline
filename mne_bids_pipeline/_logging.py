"""Logging."""
import logging
import os
import pathlib
from typing import Optional, Union

from ._typing import PathLike, LogKwargsT

logger = logging.getLogger('mne-bids-pipeline')


def gen_log_kwargs(
    message: str,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    emoji: str = '⏳️',
    script_path: Optional[PathLike] = None,
) -> LogKwargsT:
    if subject is not None:
        subject = f' sub-{subject}'
    if session is not None:
        session = f' ses-{session}'
    if run is not None:
        run = f' run-{run}'

    message = f' {message}'

    script_path = pathlib.Path(os.environ['MNE_BIDS_STUDY_SCRIPT_PATH'])
    step_name = f'{script_path.parent.name}/{script_path.stem}'

    # Choose some to be our standards
    emoji = dict(
        cache='✅',
        skip='⏩',
    ).get(emoji, emoji)
    extra = {
        'step': f'{emoji} {step_name}',
        'box': '│ ',
    }
    if subject:
        extra['subject'] = subject
    if session:
        extra['session'] = session
    if run:
        extra['run'] = run

    kwargs: LogKwargsT = {
        'msg': message,
        'extra': extra,
    }
    return kwargs
