import fire
import sys
import os
import subprocess
import pathlib
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', level='DEBUG',
                    logger=logger)

PYTHON = sys.executable
STUDY_TEMPLATE_DIR = pathlib.Path(__file__).parent
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

REPORT_SCRIPTS = ('99-make_reports.py')


def _run_script(script, config, root_dir):
    logger.info(f'Running: {script}')
    if not config:
        logger.critical('Please specify a Study Template configuration via '
                        '--config=/path/to/config.py')
        sys.exit(1)

    script_path = (STUDY_TEMPLATE_DIR / script).name
    cmd = [PYTHON, script_path]

    env = os.environ.copy()
    env['MNE_BIDS_STUDY_CONFIG'] = pathlib.Path(config).expanduser()
    if root_dir:
        env['BIDS_ROOT'] = pathlib.Path(root_dir).expanduser()

    completed_process = subprocess.run(cmd, env=env,
                                       stdout=sys.stdout,
                                       stderr=sys.stdout)
    if completed_process.returncode != 0:
        logger.critical(f'Encountered an error while running: {script}. '
                        f'Aborting.')
        sys.exit(1)


def sensor(config=None, root_dir=None):
    """Run sensor-level processing & analysis."""
    for script in SENSOR_SCRIPTS:
        _run_script(script, config, root_dir)


def source(config=None, root_dir=None):
    """Run source-level processing & analysis."""
    for script in SOURCE_SCRIPTS:
        _run_script(script, config, root_dir)


def report(config=None, root_dir=None):
    """Create processing & analysis reports."""
    for script in REPORT_SCRIPTS:
        _run_script(script, config, root_dir)


def all(config=None, root_dir=None):
    """
    Run sensor and source level processing & analysis, and create reports.
    """
    sensor(config)
    source(config)
    report(config)


if __name__ == '__main__':
    fire.Fire(dict(sensor=sensor,
                   source=source,
                   report=report,
                   all=all))
