"""Logging."""
import logging
import os
import pathlib
from typing import Optional, Union

import coloredlogs

from ._typing import LogKwargsT

logger = logging.getLogger('mne-bids-pipeline')

log_level_styles = {
    'info': {
        'bright': True,
        'bold': True,
    }
}
log_field_styles = {
    'asctime': {
        'color': 'green'
    },
    'step': {
        'color': 'cyan',
        'bold': True,
        'bright': True,
    },
    'msg': {
        'color': 'cyan',
        'bold': True,
        'bright': True,
    },
    'box': {
        'color': 'cyan',
        'bold': True,
        'bright': True,
    },
}
log_fmt = '[%(asctime)s] %(box)s%(step)s%(message)s'


class LogFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'step'):
            record.step = ''
        if not hasattr(record, 'subject'):
            record.subject = ''
        if not hasattr(record, 'session'):
            record.session = ''
        if not hasattr(record, 'run'):
            record.run = ''
        if not hasattr(record, 'box'):
            record.box = '╶╴'

        return True


logger.addFilter(LogFilter())

coloredlogs.install(
    fmt=log_fmt, level='info', logger=logger,
    level_styles=log_level_styles, field_styles=log_field_styles,
)


def gen_log_kwargs(
    message: str,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    emoji: str = '⏳️',
) -> LogKwargsT:
    if subject is not None:
        subject = f' sub-{subject}'
    if session is not None:
        session = f' ses-{session}'
    if run is not None:
        run = f' run-{run}'

    message = f' {message}'

    script_path = os.environ.get('MNE_BIDS_STUDY_SCRIPT_PATH')
    if script_path:
        script_path = pathlib.Path(script_path)
        step_name = f'{script_path.parent.name}/{script_path.stem}'
    else:  # We probably got called from run.py, not from a processing script
        step_name = ''

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
