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
                  '04-group_average_sensors.py')

SOURCE_SCRIPTS = ('01-make_forward.py',
                  '02-make_cov.py',
                  '03-make_inverse.py',
                  '04-group_average_source.py')

REPORT_SCRIPTS = ('01-make_reports.py',)

ALL_SCRIPTS = (PREPROCESSING_SCRIPTS + SENSOR_SCRIPTS + SOURCE_SCRIPTS + 
               REPORT_SCRIPTS)


def _run_script(script, config, root_dir, subject, session, task, run):
    logger.info(f'Now running: {script}')

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

    # Keep old implementation using run_module here in case we decide to
    # switch back …
    # module_name = script.replace('.py', '')
    # runpy.run_module(mod_name=module_name, run_name='__main__')

    script_path = pathlib.Path(__file__).parent / 'scripts'
    if script in PREPROCESSING_SCRIPTS:
        script_path = script_path / '01-preprocessing' / script
    elif script in SENSOR_SCRIPTS:
        script_path = script_path / '02-sensor_level' / script
    elif script in SOURCE_SCRIPTS:
        script_path = script_path / '03-source_level' / script
    elif script in REPORT_SCRIPTS:
        script_path = script_path / '04-reports' / script

    runpy.run_path(script_path, run_name='__main__')
    logger.info(f'Successfully finished running: {script}')


def process(steps: Union[Literal['sensors', 'source', 'report', 'all'], str],
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
        filename extension, separayed by a '/'. For exmaple, to run ICA, you
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

    if group == 'preprocessing':
        scripts = PREPROCESSING_SCRIPTS
    elif group == 'sensor':
        scripts = SENSOR_SCRIPTS
    elif group == 'source':
        scripts = SOURCE_SCRIPTS
    elif group == 'report':
        scripts = REPORT_SCRIPTS
    elif group == 'all':
        scripts = ALL_SCRIPTS
    else:
        raise ValueError(f'Invalid steps requested: {steps}')

    if step is not None:
        for idx, script in enumerate(scripts):
            if step in script:
                scripts = script
                break
            elif idx == len(scripts) - 1:
                # We've iterated over all scripts, but none matched!
                raise ValueError(f'Invalid steps requested: {group/steps}')

    if isinstance(scripts, str):
        scripts = (scripts,)

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

    for script in scripts:
        _run_script(script, config, root_dir, subject, session, task, run)


if __name__ == '__main__':
    fire.Fire(process)
