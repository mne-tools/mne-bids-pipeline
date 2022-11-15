"""Logging."""
import inspect
import logging
from typing import Optional, Union

import coloredlogs

from .typing import LogKwargsT

logger = logging.getLogger('mne-bids-pipeline')

log_level_styles = {
    'info': {
        'bright': True,
        'bold': True,
    },
    'warning': {
        'color': 202,
        'bold': True
    },
    'error': {
        'background': 'red',
        'bold': True
    },
    'critical': {
        'background': 'red',
        'bold': True
    },
}
log_field_styles = {
    'asctime': {
        'color': 'green'
    },
    'box': {
        'color': 'cyan',
        'bold': True,
        'bright': True,
    },
    'step': {
        'color': 'cyan',
        'bold': True,
        'bright': True,
    },
    'msg': {
        'color': 'cyan',
        'bold': True,
        'bright': False,
    },
    'subject': {
        'color': 'cyan',
        'bright': True,
        'bold': True,
    },
    'session': {
        'color': 'cyan',
        'bold': True,
        'bright': False,
    },
    'run': {
        'color': 'cyan',
        'bold': True,
        'bright': False,
    },
    'message': {
        'color': 'white',
        'bold': True,
        'bright': True,
    },
}
log_fmt = (
    '[%(asctime)s] %(box)s%(step)s%(subject)s%(session)s%(run)s%(message)s'
)
log_date_fmt = '%H:%M:%S'


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
            record.box = '│ '

        return True


logger.addFilter(LogFilter())


def _install_logs():
    coloredlogs.DEFAULT_DATE_FORMAT = log_date_fmt
    coloredlogs.install(
        fmt=log_fmt, level='info', logger=logger,
        level_styles=log_level_styles, field_styles=log_field_styles,
        date_fmt=log_date_fmt,
    )


_install_logs()


def gen_log_kwargs(
    message: str,
    *,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    step: Optional[str] = None,
    emoji: str = '⏳️',
    box: str = '│ ',
) -> LogKwargsT:
    from ._run import _get_step_path, _short_step_path
    # Try to figure these out
    stack = inspect.stack()
    up_locals = stack[1].frame.f_locals
    if subject is None:
        subject = up_locals.get('subject', None)
    if session is None:
        session = up_locals.get('session', None)
    if run is None:
        run = up_locals.get('run', None)
    if step is None:
        step_path = _get_step_path(stack)
        if step_path:
            step = _short_step_path(_get_step_path())
        else:
            step = ''

    # Do some nice formatting
    if subject is not None:
        subject = f' sub-{subject}'
    if session is not None:
        session = f' ses-{session}'
    if run is not None:
        run = f' run-{run}'
    if step != '':
        # need an extra space
        message = f' {message}'

    # Choose some to be our standards
    emoji = dict(
        cache='✅',
        skip='⏩',
        override='❌',
    ).get(emoji, emoji)
    extra = {
        'step': f'{emoji} {step}',
        'box': box,
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
