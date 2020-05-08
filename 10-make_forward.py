"""
====================
12. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import glob
import os.path as op
import itertools
import logging

import mne

from mne.parallel import parallel_func
from mne_bids import make_bids_basename, get_head_mri_trans
from mne_bids.read import reader as mne_bids_readers

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def run_forward(subject, session=None):
    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
    fname_evoked = \
        op.join(fpath_deriv, bids_basename + '-ave.fif')

    fname_trans = \
        op.join(fpath_deriv, 'sub-{}'.format(subject) + '-trans.fif')

    fname_fwd = \
        op.join(fpath_deriv, bids_basename + '-fwd.fif')

    msg = f'Input: {fname_evoked}, Output: {fname_fwd}'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    # Find the raw data file
    # XXX : maybe simplify
    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=config.get_runs()[0],
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    data_dir = op.join(config.bids_root, subject_path)
    search_str = (op.join(data_dir, bids_basename) + '_' +
                  config.get_kind() + '*')
    fnames = sorted(glob.glob(search_str))
    fnames = [f for f in fnames
              if op.splitext(f)[1] in mne_bids_readers]

    if len(fnames) >= 1:
        bids_fname = fnames[0]
    elif len(fnames) == 0:
        raise ValueError('Could not find input data file matching: '
                         '"{}"'.format(search_str))

    bids_fname = op.basename(bids_fname)
    trans = get_head_mri_trans(bids_fname=bids_fname,
                               bids_root=config.bids_root)

    mne.write_trans(fname_trans, trans)

    src = mne.setup_source_space(subject, spacing=config.spacing,
                                 subjects_dir=config.get_subjects_dir(),
                                 add_dist=False)

    evoked = mne.read_evokeds(fname_evoked, condition=0)

    # Here we only use 3-layers BEM only if EEG is available.
    if config.get_kind() == 'eeg':
        model = mne.make_bem_model(subject, ico=4,
                                   conductivity=(0.3, 0.006, 0.3),
                                   subjects_dir=config.get_subjects_dir())
    else:
        model = mne.make_bem_model(subject, ico=4, conductivity=(0.3,),
                                   subjects_dir=config.get_subjects_dir())

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
