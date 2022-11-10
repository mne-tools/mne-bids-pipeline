#!/usr/bin/env python

import argparse
import os
import pathlib
from textwrap import dedent
import time
from typing import List
from types import ModuleType

import numpy as np

from ._config_utils import _get_script_modules
from ._config_import import _import_config
from ._config_template import create_template_config
from ._logging import logger


def main():
    from . import __version__
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('config', nargs='?', default=None)
    parser.add_argument(
        '--config', dest='config_switch', default=None, metavar='FILE',
        help='The path of the pipeline configuration file to use.')
    parser.add_argument(
        '--create-config', dest='create_config', default=None, metavar='FILE',
        help='Create a template configuration file with the specified name. '
             'If specified, all other parameters will be ignored.'
    ),
    parser.add_argument(
        '--steps', dest='steps', default='all',
        help=dedent("""\
        The processing steps to run.
        Can either be one of the processing groups 'preprocessing', sensor',
        'source', 'report',  or 'all',  or the name of a processing group plus
        the desired script sans the step number and
        filename extension, separated by a '/'. For example, to run ICA, you
        would pass 'sensor/run_ica`. If unspecified, will run all processing
        steps. Can also be a tuple of steps."""))
    parser.add_argument(
        '--root-dir', dest='root_dir', default=None,
        help="BIDS root directory of the data to process.")
    parser.add_argument(
        '--subject', dest='subject', default=None,
        help="The subject to process.")
    parser.add_argument(
        '--session', dest='session', default=None,
        help="The session to process.")
    parser.add_argument(
        '--task', dest='task', default=None,
        help="The task to process.")
    parser.add_argument(
        '--run', dest='run', default=None,
        help="The run to process.")
    parser.add_argument(
        '--n_jobs', dest='n_jobs', type=int, default=None,
        help="The number of parallel processes to execute.")
    parser.add_argument(
        '--interactive', dest='interactive', action='store_true',
        help="Enable interactive mode.")
    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        help="Enable debugging on error.")
    parser.add_argument(
        '--no-cache', dest='no_cache', action='store_true',
        help='Disable caching of intermediate results.')
    options = parser.parse_args()

    if options.create_config is not None:
        target_path = pathlib.Path(options.create_config)
        create_template_config(target_path=target_path, overwrite=False)
        return

    config = options.config
    config_switch = options.config_switch
    bad = False
    if config is None:
        if config_switch is None:
            bad = 'neither was provided'
        else:
            config = config_switch
    elif config_switch is not None:
        bad = 'both were provided'
    if bad:
        parser.error(
            '❌ You must specify a configuration file either as a single '
            f'argument or with --config, but {bad}.')
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
    os.environ['MNE_BIDS_STUDY_CONFIG'] = config
    if root_dir:
        os.environ['BIDS_ROOT'] = str(pathlib.Path(root_dir).expanduser())
    if subject:
        os.environ['MNE_BIDS_STUDY_SUBJECT'] = subject
    if session:
        os.environ['MNE_BIDS_STUDY_SESSION'] = session
    if task:
        os.environ['MNE_BIDS_STUDY_TASK'] = task
    if run:
        os.environ['MNE_BIDS_STUDY_RUN'] = run
    if interactive:
        os.environ['MNE_BIDS_STUDY_INTERACTIVE'] = str(int(interactive))
    if n_jobs:
        os.environ['MNE_BIDS_STUDY_NJOBS'] = str(n_jobs)
    if on_error:
        os.environ['MNE_BIDS_STUDY_ON_ERROR'] = on_error
    if cache:
        os.environ['MNE_BIDS_STUDY_USE_CACHE'] = cache

    script_modules: List[ModuleType] = []
    SCRIPT_MODULES = _get_script_modules()
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

    config_imported = _import_config(log=True)
    for script_module in script_modules:
        this_name = script_module.__name__.split('.', maxsplit=1)[-1]
        this_name = this_name.replace('.', '/')
        extra = dict(box='┌╴', step=f'🚀 {this_name} ')
        start = time.time()
        logger.info('Now running  👇', extra=extra)
        script_module.main(config=config_imported)
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
