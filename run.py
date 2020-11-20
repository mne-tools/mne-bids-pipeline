#!/usr/bin/env python 

import fire
import os
import runpy
import signal
import atexit
import pathlib
import logging
import coloredlogs
from typing import Union, Optional
try:
    from typing import Literal
except ImportError:  # Python <3.8
    from typing_extensions import Literal

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', logger=logger)

PREPROCESSING_SCRIPTS = ('01-import_and_maxfilter.py',
                         '02-frequency_filter.py',
                         '03-make_epochs.py',
                         '04a-run_ica.py',
                         '04b-run_ssp.py',
                         '05a-apply_ica.py',
                         '05b-apply_ssp.py')

SENSOR_SCRIPTS = ('01-make_evoked.py',
                  '02-sliding_estimator.py',
                  '03-time_frequency.py',
                  '04-group_average.py')

SOURCE_SCRIPTS = ('01-make_forward.py',
                  '02-make_cov.py',
                  '03-make_inverse.py',
                  '04-group_average.py')

REPORT_SCRIPTS = ('01-make_reports.py',)

SCRIPT_BASE_DIR = pathlib.Path(__file__).parent / 'scripts'

SCRIPT_PATHS = {
    'preprocessing': [SCRIPT_BASE_DIR / 'preprocessing' / s
                      for s in PREPROCESSING_SCRIPTS],
    'sensor': [SCRIPT_BASE_DIR / 'sensor' / s
               for s in SENSOR_SCRIPTS],
    'source': [SCRIPT_BASE_DIR / 'source' / s
               for s in SOURCE_SCRIPTS],
    'report': [SCRIPT_BASE_DIR / 'report' / s
               for s in REPORT_SCRIPTS]
}
SCRIPT_PATHS['all'] = (SCRIPT_PATHS['preprocessing'] +
                       SCRIPT_PATHS['sensor'] + SCRIPT_PATHS['source'] +
                       SCRIPT_PATHS['report'])


def _run_script(script_path, config, root_dir, subject, session, task, run):
    # It's okay to fiddle with the environment variables here as process()
    # has set up some handlers to reset the environment to its previous state
    # upon exit.
    env = os.environ
    env['MNE_BIDS_STUDY_CONFIG'] = str(pathlib.Path(config).expanduser())

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

    runpy.run_path(script_path, run_name='__main__')


def process(steps: Union[Literal['sensor', 'source', 'report', 'all'], str],
            *,
            config: Union[str, pathlib.Path],
            root_dir: Optional[Union[str, pathlib.Path]] = None,
            subject: Optional[str] = None,
            session: Optional[str] = None,
            task: Optional[str] = None,
            run: Optional[str] = None):
    """Run the Study Template.

    Parameters
    ----------
    steps
        The processing steps to run.
        Can either be one of the processing groups 'preprocessing', sensor',
        'source', 'report',  or 'all',  or the name of a processing group plus
        the desired script sans the step number and
        filename extension, separated by a '/'. For exmaple, to run ICA, you
        would pass 'sensor/run_ica`.
    config
        The path of the Study Template configuration file to use.
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
    """
    if '/' in steps:
        group, step = steps.split('/')
    else:
        group, step = steps, None

    if group not in SCRIPT_PATHS.keys():
        raise ValueError(f'Invalid steps requested: {steps}')

    if step is None:
        # User specified `sensors`, `source`, or similar
        script_paths = SCRIPT_PATHS[group]
    else:
        # User specified 'group/step'
        for idx, script_path in enumerate(SCRIPT_PATHS[group]):
            if step in str(script_path):
                script_paths = (script_path,)
                break
            if idx == len(SCRIPT_PATHS[group]) - 1:
                # We've iterated over all scripts, but none matched!
                raise ValueError(f'Invalid steps requested: {group/steps}')

    # Ensure we will restore the original environment variables in most cases
    # upon exit.
    env_orig = os.environ.copy()

    def _restore_env():
        os.environ.update(env_orig)

    signals = (
        signal.SIGINT,  # Ctrl-C
        signal.SIGTERM  # Sent by kill command
    )
    for s in signals:
        signal.signal(s, _restore_env)
    atexit.register(_restore_env)

    for script_path in script_paths:
        step_name = script_path.name.replace('.py', '')[3:]
        logger.info(f'Now running: {step_name}')
        _run_script(script_path, config, root_dir, subject, session, task, run)
        logger.info(f'Successfully finished running: {step_name}')


if __name__ == '__main__':
    fire.Fire(process)
