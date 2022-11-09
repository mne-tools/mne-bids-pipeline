#!/usr/bin/env python

import optparse
import os
import sys
import pathlib
import logging
from textwrap import dedent
import time
from typing import Union, Optional, Tuple, List, Dict
from types import ModuleType

import coloredlogs
import numpy as np


# Ensure that the "scripts" that we import from is the correct one
sys.path.insert(0, str(pathlib.Path(__file__).parent))


logger = logging.getLogger(__name__)

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

PathLike = Union[str, pathlib.Path]


def _get_script_modules(
    *,
    config: PathLike,
    root_dir: Optional[PathLike] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    interactive: Optional[str] = None,
    n_jobs: Optional[str] = None,
    on_error: Optional[str] = None,
    cache: Optional[str] = None,
) -> Dict[str, Tuple[ModuleType]]:
    env = os.environ
    env['MNE_BIDS_STUDY_CONFIG'] = config

    if root_dir:
        env['BIDS_ROOT'] = str(pathlib.Path(root_dir).expanduser())

    if task:
        env['MNE_BIDS_STUDY_TASK'] = task

    if session:
        env['MNE_BIDS_STUDY_SESSION'] = session

    if run:
        env['MNE_BIDS_STUDY_RUN'] = run

    if subject:
        env['MNE_BIDS_STUDY_SUBJECT'] = subject

    if interactive:
        env['MNE_BIDS_STUDY_INTERACTIVE'] = interactive

    if n_jobs:
        env['MNE_BIDS_STUDY_NJOBS'] = n_jobs

    if on_error:
        env['MNE_BIDS_STUDY_ON_ERROR'] = on_error

    if cache:
        env['MNE_BIDS_STUDY_USE_CACHE'] = cache

    from .scripts import init
    from .scripts import preprocessing
    from .scripts import sensor
    from .scripts import source
    from .scripts import report
    from .scripts import freesurfer

    INIT_SCRIPTS = init.SCRIPTS
    PREPROCESSING_SCRIPTS = preprocessing.SCRIPTS
    SENSOR_SCRIPTS = sensor.SCRIPTS
    SOURCE_SCRIPTS = source.SCRIPTS
    REPORT_SCRIPTS = report.SCRIPTS
    FREESURFER_SCRIPTS = freesurfer.SCRIPTS

    SCRIPT_MODULES = {
        'init': INIT_SCRIPTS,
        'freesurfer': FREESURFER_SCRIPTS,
        'preprocessing': PREPROCESSING_SCRIPTS,
        'sensor': SENSOR_SCRIPTS,
        'source': SOURCE_SCRIPTS,
        'report': REPORT_SCRIPTS,
    }

    # Do not include the FreeSurfer scripts in "all" – we don't intend to run
    # recon-all by default!
    SCRIPT_MODULES['all'] = (
        SCRIPT_MODULES['init'] +
        SCRIPT_MODULES['preprocessing'] +
        SCRIPT_MODULES['sensor'] +
        SCRIPT_MODULES['source'] +
        SCRIPT_MODULES['report']
    )

    return SCRIPT_MODULES


def main():
    from . import __version__
    parser = optparse.OptionParser(version=f'%prog {__version__}')
    parser.add_option(
        '-c', '--config', dest='config', default=None, metavar='FILE',
        help='The path of the pipeline configuration file to use.')
    parser.add_option(
        '--steps', dest='steps', default='all',
        help=dedent("""\
        The processing steps to run.
        Can either be one of the processing groups 'preprocessing', sensor',
        'source', 'report',  or 'all',  or the name of a processing group plus
        the desired script sans the step number and
        filename extension, separated by a '/'. For example, to run ICA, you
        would pass 'sensor/run_ica`. If unspecified, will run all processing
        steps. Can also be a tuple of steps."""))
    parser.add_option(
        '--root-dir', dest='root_dir', default=None,
        help="BIDS root directory of the data to process.")
    parser.add_option(
        '--subject', dest='subject', default=None,
        help="The subject to process.")
    parser.add_option(
        '--session', dest='session', default=None,
        help="The session to process.")
    parser.add_option(
        '--task', dest='task', default=None,
        help="The task to process.")
    parser.add_option(
        '--run', dest='run', default=None,
        help="The run to process.")
    parser.add_option(
        '--n_jobs', dest='n_jobs', type='int', default=None,
        help="The number of parallel processes to execute.")
    parser.add_option(
        '--interactive', dest='interactive', action='store_true',
        help="Enable interactive mode.")
    parser.add_option(
        '--debug', dest='debug', action='store_true',
        help="Enable debugging on error.")
    parser.add_option(
        '--no-cache', dest='no_cache', action='store_true',
        help='Disable caching of intermediate results.')
    options, args = parser.parse_args()
    config = options.config
    bad_msg = (
        'You must specify a configuration file as a single argument '
        'or with --config.'
    )
    if config is None:
        if len(args) == 1:
            config = args[0]
        else:
            raise ValueError(bad_msg)
    elif len(args):
        raise ValueError(bad_msg)
    steps = options.steps
    root_dir = options.root_dir
    subject, session = options.subject, options.session
    task, run = options.task, options.run
    n_jobs = options.n_jobs
    interactive, debug = options.interactive, options.debug
    cache = not options.no_cache

    if isinstance(steps, str) and ',' in steps:
        # Work around limitation in Fire: --steps=foo,bar/baz won't produce a
        # tuple ('foo', 'bar/baz'), but a string 'foo,bar/baz'.
        steps = tuple(steps.split(','))
    elif isinstance(steps, str):
        steps = (steps,)

    if n_jobs is not None:
        n_jobs = str(n_jobs)
    on_error = 'debug' if debug else None
    cache = '1' if cache else '0'

    processing_stages = []
    processing_steps = []
    for steps_ in steps:
        if '/' in steps_:
            stage, step = steps_.split('/')
            processing_stages.append(stage)
            processing_steps.append(step)
        else:
            # User specified "sensor", "preprocessing" or similar, but without
            # any further grouping.
            processing_stages.append(steps_)
            processing_steps.append(None)

    config = str(pathlib.Path(config).expanduser())
    SCRIPT_MODULES = _get_script_modules(
        config=config,
        root_dir=root_dir,
        subject=subject,
        session=session,
        task=task,
        run=run,
        interactive=interactive,
        n_jobs=n_jobs,
        on_error=on_error,
        cache=cache,
    )

    script_modules: List[ModuleType] = []
    for stage, step in zip(processing_stages, processing_steps):
        if stage not in SCRIPT_MODULES.keys():
            raise ValueError(
                f"Invalid step requested: '{stage}'. "
                f'It should be one of {list(SCRIPT_MODULES.keys())}.'
            )

        if step is None:
            # User specified `sensors`, `source`, or similar
            script_modules.extend(SCRIPT_MODULES[stage])
        else:
            # User specified 'stage/step'
            for script_module in SCRIPT_MODULES[stage]:
                script_name = pathlib.Path(script_module.__file__).name
                if step in script_name:
                    script_modules.append(script_module)
                    break
            else:
                # We've iterated over all scripts, but none matched!
                raise ValueError(f'Invalid steps requested: {stage}/{step}')

    if processing_stages[0] != 'all':
        # Always run the directory initialization scripts, but skip for 'all',
        # because it already includes them – and we want to avoid running
        # them twice.
        script_modules = [*SCRIPT_MODULES['init'], *script_modules]

    logger.info(
        "👋 Welcome aboard the MNE BIDS Pipeline!"
    )
    logger.info(
        f"🧾 Using configuration: {config}"
    )

    for script_module in script_modules:
        this_name = script_module.__name__.split('.', maxsplit=1)[-1]
        this_name = this_name.replace('.', '/')
        extra = dict(box='┌╴', step=f'🚀 {this_name} ')
        start = time.time()
        logger.info('Now running  👇', extra=extra)
        script_module.main()
        extra = dict(box='└╴', step=f'🎉 {this_name} ')
        elapsed = time.time() - start
        hours, remainder = divmod(elapsed, 3600)
        hours = int(hours)
        minutes, seconds = divmod(remainder, 60)
        minutes = int(minutes)
        seconds = int(np.ceil(seconds))  # always take full seconds
        elapsed = f'{seconds}s'
        if minutes:
            elapsed = f'{minutes}m {elapsed}'
        if hours:
            elapsed = f'{hours}h {elapsed}'
        logger.info(f'Done running 👆 [{elapsed}]', extra=extra)


if __name__ == '__main__':
    main()
