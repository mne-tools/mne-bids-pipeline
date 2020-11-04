import fire
import sys
import os
import runpy
import signal
import atexit
import pathlib
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', logger=logger)

SENSOR_SCRIPTS = ('01-import_and_maxfilter.py',
                  '02-frequency_filter.py',
                  '03-make_epochs.py',
                  '04a-run_ica.py',
                  '04b-run_ssp.py',
                  '05a-apply_ica.py',
                  '05b-apply_ssp.py',
                  '06-make_evoked.py',
                  '07-sliding_estimator.py',
                  '08-time_frequency.py',
                  '09-group_average_sensors.py')

SOURCE_SCRIPTS = ('10-make_forward.py',
                  '11-make_cov.py',
                  '12-make_inverse.py',
                  '13-group_average_source.py')

REPORT_SCRIPTS = ('99-make_reports.py',)

ALL_SCRIPTS = SENSOR_SCRIPTS + SOURCE_SCRIPTS + REPORT_SCRIPTS


def _run_script(script, config, root_dir):
    logger.info(f'Now running: {script}')
    if not config:
        logger.critical('Please specify a Study Template configuration '
                        'via --config=/path/to/config.py')
        sys.exit(1)

    env = os.environ
    env['MNE_BIDS_STUDY_CONFIG'] = str(pathlib.Path(config).expanduser())
    if root_dir:
        env['BIDS_ROOT'] = str(pathlib.Path(root_dir).expanduser())

    module_name = script.replace('.py', '')
    runpy.run_module(mod_name=module_name, run_name='__main__')

    logger.info(f'Successfully finished running: {script}')


def process(steps=None, config=None, root_dir=None):
    if not steps:
        logger.critical('Please specify which processing step(s) to run via '
                        '--steps=...')
        sys.exit(1)

    if steps == 'sensor':
        scripts = SENSOR_SCRIPTS
    elif steps == 'source':
        scripts = SOURCE_SCRIPTS
    elif steps == 'report':
        scripts = REPORT_SCRIPTS
    elif steps == 'all':
        scripts = ALL_SCRIPTS
    else:
        scripts = ALL_SCRIPTS[steps-1]

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
        _run_script(script, config, root_dir)


if __name__ == '__main__':
    fire.Fire(dict(process=process))
