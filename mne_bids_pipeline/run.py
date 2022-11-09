#!/usr/bin/env python

import os
import pathlib
import sys
import time
from typing import Union, Optional, Tuple, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from types import ModuleType

import fire
import numpy as np

from ._config_utils import _get_script_modules, _import_config
from ._logging import logger
from ._typing import PathLike


Step_T = Union[Literal['preprocessing', 'sensor', 'source', 'report', 'all',
                       'freesurfer'], str]
Steps_T = Union[Step_T, Tuple[Step_T]]


def process(
    *,
    config: PathLike,
    steps: Optional[Steps_T] = None,
    root_dir: Optional[PathLike] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    interactive: Optional[str] = None,
    n_jobs: Optional[str] = None,
    debug: Optional[str] = None,
    cache: Optional[str] = None,
    **kwargs: Optional[dict],
):
    """Run the BIDS pipeline.

    Parameters
    ----------
    config
        The path of the pipeline configuration file to use.
    steps
        The processing steps to run.
        Can either be one of the processing groups 'preprocessing', sensor',
        'source', 'report',  or 'all',  or the name of a processing group plus
        the desired script sans the step number and
        filename extension, separated by a '/'. For example, to run ICA, you
        would pass 'sensor/run_ica`. If unspecified, will run all processing
        steps. Can also be a tuple of steps.
    root_dir
        BIDS root directory of the data to process.
    subject
        The subject to process.
    session
        The session to process.
    task
        The task to process.
    run
        The run to process.
    interactive
        Whether or not to enable "interactive" mode.
    n_jobs
        The number of parallel processes to execute.
    debug
        Whether or not to force on_error='debug'.
    cache
        Whether or not to use caching.
    **kwargs
        Should not be used. Only used to detect invalid arguments.
    """
    if kwargs:
        raise ValueError(f"Unknown argument(s) to run.py: {list(kwargs)}")
    if steps is None:
        steps = ('all',)
    elif isinstance(steps, str) and ',' in steps:
        # Work around limitation in Fire: --steps=foo,bar/baz won't produce a
        # tuple ('foo', 'bar/baz'), but a string 'foo,bar/baz'.
        steps = tuple(steps.split(','))
    elif isinstance(steps, str):
        steps = (steps,)

    assert isinstance(steps, tuple)

    # Fire auto-detects the input parameter values, but this means e.g. that an
    # unquoted subject ID "123" will be treated as an int. To avoid this, cast
    # input values to str.
    # Note that parameters values starting with a zero padding are
    # automatically treated as strings by Fire, and are not affected by the
    # following block of type casts.
    if subject is not None:
        subject = str(subject)
    if session is not None:
        session = str(session)
    if run is not None:
        run = str(run)
    if task is not None:
        task = str(task)
    if interactive is not None:
        interactive = '1' if interactive in ['1', 'True', True] else '0'
    if n_jobs is not None:
        n_jobs = str(n_jobs)
    on_error = 'debug' if debug is not None else debug
    cache = '1' if cache != 0 else '0'

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
        os.environ['MNE_BIDS_STUDY_INTERACTIVE'] = interactive
    if n_jobs:
        os.environ['MNE_BIDS_STUDY_NJOBS'] = n_jobs
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

    config_imported = _import_config()
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


def main():
    # Fire does not seem to detect a "--help" in all locations, so let's do it
    # manually.
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv = sys.argv[:1] + ['--help']
    fire.Fire(process)
