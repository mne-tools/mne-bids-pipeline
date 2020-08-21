"""
====================
12. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import os.path as op
import itertools
import logging

import mne

from mne.parallel import parallel_func
from mne_bids import make_bids_basename, get_head_mri_trans

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def run_forward(subject, session=None):
    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       recording=config.rec,
                                       space=config.space,
                                       prefix=deriv_path)

    fname_evoked = bids_basename.copy().update(kind='ave', extension='.fif')
    fname_trans = bids_basename.copy().update(kind='trans', extension='.fif')
    fname_fwd = bids_basename.copy().update(kind='fwd', extension='.fif')

    msg = f'Input: {fname_evoked}, Output: {fname_fwd}'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    # Find the raw data file
    trans = get_head_mri_trans(
        bids_basename=(bids_basename.copy()
                       .update(run=config.get_runs()[0], prefix=None)),
        bids_root=config.bids_root)

    mne.write_trans(fname_trans, trans)

    src = mne.setup_source_space(subject, spacing=config.spacing,
                                 subjects_dir=config.get_fs_subjects_dir(),
                                 add_dist=False)

    evoked = mne.read_evokeds(fname_evoked, condition=0)

    # Here we only use 3-layers BEM only if EEG is available.
    if 'eeg' in config.ch_types:
        model = mne.make_bem_model(subject, ico=4,
                                   conductivity=(0.3, 0.006, 0.3),
                                   subjects_dir=config.get_fs_subjects_dir())
    else:
        model = mne.make_bem_model(subject, ico=4, conductivity=(0.3,),
                                   subjects_dir=config.get_fs_subjects_dir())

    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(evoked.info, trans, src, bem,
                                    mindist=config.mindist)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


def main():
    """Run forward."""
    msg = 'Running Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))

    parallel, run_func, _ = parallel_func(run_forward, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))


if __name__ == '__main__':
    main()
