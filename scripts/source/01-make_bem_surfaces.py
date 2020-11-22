"""
===================
Create BEM surfaces
===================

Extract the BEM surfaces using the watershed algorithm. This is required to
produce the forward solution in the next step.
"""

import logging
from pathlib import Path

import mne
from mne.parallel import parallel_func

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def make_bem(subject):
    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()
    watershed_dir = Path(fs_subjects_dir) / fs_subject / 'bem' / 'watershed'
    show = True if config.interactive else False

    if watershed_dir.exists():
        msg = 'Found existing BEM surfaces. '
        if config.recreate_bem_surfaces:
            msg += 'Overwriting as requested in configuration.'
            logger.info(gen_log_message(step=10, message=msg))
        else:
            msg = 'Skipping surface extraction as requested in configuration.'
            logger.info(gen_log_message(step=10, message=msg))
            return

    msg = 'Creating BEM surfaces using watershed algorithm'
    logger.info(gen_log_message(step=10, message=msg))

    mne.bem.make_watershed_bem(subject=fs_subject,
                               subjects_dir=fs_subjects_dir,
                               copy=True,  # XXX Revise!
                               overwrite=True,
                               show=show)


def main():
    """Run BEM surface extraction."""
    msg = 'Running Step 10: Create BEM surfaces'
    logger.info(gen_log_message(step=10, message=msg))

    parallel, run_func, _ = parallel_func(make_bem, n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.get_subjects())

    msg = 'Completed Step 10: Create BEM surfaces'
    logger.info(gen_log_message(step=10, message=msg))


if __name__ == '__main__':
    main()
