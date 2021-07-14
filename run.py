#!/usr/bin/env python

import os
import sys
import runpy
import pathlib
import logging
from typing import Union, Optional, Tuple
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import fire
import coloredlogs

logger = logging.getLogger(__name__)

log_level_styles = {
    'info': {
        'bright': True,
        'bold': True
    }
}
log_fmt = '\n%(message)s'
coloredlogs.install(
    fmt=log_fmt,
    level_styles=log_level_styles,
    logger=logger
)

PathLike = Union[str, pathlib.Path]


INIT_SCRIPTS = ('00-init_derivatives_dir.py',)

PREPROCESSING_SCRIPTS = (
    '01-maxfilter.py',
    '02-frequency_filter.py',
    '03-make_epochs.py',
    '04a-run_ica.py',
    '04b-run_ssp.py',
    '05a-apply_ica.py',
    '05b-apply_ssp.py',
    '06-ptp_reject.py'
)

SENSOR_SCRIPTS = (
    '01-make_evoked.py',
    '02-sliding_estimator.py',
    '03-time_frequency.py',
    '04-group_average.py'
)

SOURCE_SCRIPTS = (
    '01-make_bem_surfaces.py',
    '02-make_forward.py',
    '03-make_cov.py',
    '04-make_inverse.py',
    '05-group_average.py'
)

REPORT_SCRIPTS = ('01-make_reports.py',)

FREESURFER_SCRIPTS = (
    '01-recon_all.py',
    '02-coreg_surfaces.py'
)

SCRIPT_BASE_DIR = pathlib.Path(__file__).parent / 'scripts'

SCRIPT_PATHS = {
    'init': [SCRIPT_BASE_DIR / 'init' / s
             for s in INIT_SCRIPTS],
    'preprocessing': [SCRIPT_BASE_DIR / 'preprocessing' / s
                      for s in PREPROCESSING_SCRIPTS],
    'sensor': [SCRIPT_BASE_DIR / 'sensor' / s
               for s in SENSOR_SCRIPTS],
    'source': [SCRIPT_BASE_DIR / 'source' / s
               for s in SOURCE_SCRIPTS],
    'report': [SCRIPT_BASE_DIR / 'report' / s
               for s in REPORT_SCRIPTS],
    'freesurfer': [SCRIPT_BASE_DIR / 'freesurfer' / s
                   for s in FREESURFER_SCRIPTS]
}

# Do not include the FreeSurfer scripts in "all" – we don't intend to run
# recon-all by default!
SCRIPT_PATHS['all'] = (SCRIPT_PATHS['init'] +
                       SCRIPT_PATHS['preprocessing'] +
                       SCRIPT_PATHS['sensor'] + SCRIPT_PATHS['source'] +
                       SCRIPT_PATHS['report'])


def _run_script(script_path, config, root_dir, subject, session, task, run,
                n_jobs):
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

    if n_jobs:
        env['MNE_BIDS_STUDY_NJOBS'] = n_jobs

    runpy.run_path(script_path, run_name='__main__')


Step_T = Union[Literal['preprocessing', 'sensor', 'source', 'report', 'all',
                       'freesurfer'], str]
Steps_T = Union[Step_T, Tuple[Step_T]]


def process(config: PathLike,
            steps: Optional[Steps_T] = None,
            root_dir: Optional[PathLike] = None,
            subject: Optional[str] = None,
            session: Optional[str] = None,
            task: Optional[str] = None,
            run: Optional[str] = None,
            n_jobs: Optional[str] = None):
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
        filename extension, separated by a '/'. For exmaple, to run ICA, you
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
    n_jobs
        The number of parallel processes to execute.
    """
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
    if n_jobs is not None:
        n_jobs = str(n_jobs)

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

    script_paths = []
    for stage, step in zip(processing_stages, processing_steps):
        if stage not in SCRIPT_PATHS.keys():
            raise ValueError(
                f"Invalid step requested: '{stage}'. "
                f'It should be one of {list(SCRIPT_PATHS.keys())}.'
            )

        if step is None:
            # User specified `sensors`, `source`, or similar
            script_paths.extend(SCRIPT_PATHS[stage])
        else:
            # User specified 'stage/step'
            for script_path in SCRIPT_PATHS[stage]:
                if step in str(script_path):
                    script_paths.append(script_path)
                    break
            else:
                # We've iterated over all scripts, but none matched!
                raise ValueError(f'Invalid steps requested: {stage}/{step}')

    if processing_stages[0] != 'all':
        # Always run the directory initialization scripts, but skip for 'all',
        # because it already includes them – and we want to avoid running
        # them twice.
        script_paths = [*SCRIPT_PATHS['init'], *script_paths]

    logger.info(
        "👋 Welcome aboard the MNE BIDS Pipeline!\n"
        "   Please fasten your seatbelt. We hope you'll enjoy your flight.\n"
    )
    for script_path in script_paths:
        step_name = f'{script_path.parent.name}/{script_path.name}'
        logger.info(f'🚀 Now running script: {step_name} 👇\n')
        _run_script(script_path, config, root_dir, subject, session, task, run,
                    n_jobs)


if __name__ == '__main__':
    fire.Fire(process)
