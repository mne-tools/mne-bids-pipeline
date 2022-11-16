import argparse
import pathlib
from textwrap import dedent
import time
from typing import List
from types import ModuleType, SimpleNamespace

import numpy as np

from ._config_utils import _get_step_modules
from ._config_import import _import_config
from ._config_template import create_template_config
from ._logging import logger, gen_log_kwargs, _install_logs
from ._run import _short_step_path


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
        the desired step sans the step number and
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

    config_path = pathlib.Path(config).expanduser().resolve(strict=True)
    overrides = SimpleNamespace()
    if root_dir:
        overrides.bids_root = \
            pathlib.Path(root_dir).expanduser().resolve(strict=True)
    if subject:
        overrides.subjects = [subject]
    if session:
        overrides.sessions = [session]
    if task:
        overrides.task = task
    if run:
        overrides.runs = run
    if interactive:
        overrides.interactive = interactive
    if n_jobs:
        overrides.N_JOBS = int(n_jobs)
    if on_error:
        overrides.on_error = on_error
    if not cache:
        overrides.memory_location = False

    step_modules: List[ModuleType] = []
    STEP_MODULES = _get_step_modules()
    for stage, step in zip(processing_stages, processing_steps):
        if stage not in STEP_MODULES.keys():
            raise ValueError(
                f"Invalid step requested: '{stage}'. "
                f'It should be one of {list(STEP_MODULES.keys())}.'
            )

        if step is None:
            # User specified `sensors`, `source`, or similar
            step_modules.extend(STEP_MODULES[stage])
        else:
            # User specified 'stage/step'
            for step_module in STEP_MODULES[stage]:
                step_name = pathlib.Path(step_module.__file__).name
                if step in step_name:
                    step_modules.append(step_module)
                    break
            else:
                # We've iterated over all steps, but none matched!
                raise ValueError(f'Invalid steps requested: {stage}/{step}')

    if processing_stages[0] != 'all':
        # Always run the directory initialization steps, but skip for 'all',
        # because it already includes them – and we want to avoid running
        # them twice.
        step_modules = [*STEP_MODULES['init'], *step_modules]

    _install_logs()
    msg = "Welcome aboard the MNE BIDS Pipeline!"
    logger.info(**gen_log_kwargs(message=msg, emoji='👋', box='╶╴', step=''))
    msg = f"Using configuration: {config}"
    logger.info(**gen_log_kwargs(message=msg, emoji='🧾', box='╶╴', step=''))

    config_imported = _import_config(
        config_path=config_path,
        overrides=overrides,
    )
    for step_module in step_modules:
        start = time.time()
        step = _short_step_path(pathlib.Path(step_module.__file__))
        msg = 'Now running  👇'
        logger.info(
            **gen_log_kwargs(message=msg, box='┌╴', emoji='🚀', step=step)
        )
        step_module.main(config=config_imported)
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
        msg = f'Done running  👆 [{elapsed}]'
        logger.info(
            **gen_log_kwargs(message=msg, box='└╴', emoji='🎉', step=step)
        )
