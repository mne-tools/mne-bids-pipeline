"""
====================
10. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import itertools
import logging

import mne

from mne.parallel import parallel_func
from mne_bids import BIDSPath, get_head_mri_trans

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def run_forward(subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    fname_evoked = bids_path.copy().update(suffix='ave')
    fname_trans = bids_path.copy().update(suffix='trans')
    fname_fwd = bids_path.copy().update(suffix='fwd')

    # Generate a head ↔ MRI transformation matrix from the
    # electrophysiological and MRI sidecar files, and save it to an MNE
    # "trans" file in the derivatives folder.
    if config.mri_t1_path_generator is None:
        t1_bids_path = None
    else:
        t1_bids_path = BIDSPath(subject=bids_path.subject,
                                session=bids_path.session,
                                root=bids_path.root)
        t1_bids_path = config.mri_t1_path_generator(t1_bids_path.copy())
        if t1_bids_path.suffix is None:
            t1_bids_path.update(suffix='T1w')
        if t1_bids_path.datatype is None:
            t1_bids_path.update(datatype='anat')

    msg = 'Estimating head ↔ MRI transform'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))

    trans = get_head_mri_trans(
        bids_path.copy().update(run=config.get_runs()[0],
                                root=config.bids_root),
        t1_bids_path=t1_bids_path)
    mne.write_trans(fname_trans, trans)

    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()

    # Create the source space.
    msg = 'Creating source space'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    src = mne.setup_source_space(subject=fs_subject,
                                 subjects_dir=fs_subjects_dir,
                                 spacing=config.spacing,
                                 add_dist=False,
                                 n_jobs=config.N_JOBS)

    # Calculate the BEM solution.
    # Here we only use a 3-layers BEM only if EEG is available.
    msg = 'Calculating BEM solution'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))

    if 'eeg' in config.ch_types:
        conductivity = (0.3, 0.006, 0.3)
    else:
        conductivity = (0.3,)

    bem_model = mne.make_bem_model(subject=fs_subject,
                                   subjects_dir=fs_subjects_dir,
                                   ico=4, conductivity=conductivity)
    bem_sol = mne.make_bem_solution(bem_model)

    # Finally, calculate and save the forward solution.
    msg = 'Calculating forward solution'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    info = mne.io.read_info(fname_evoked)
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem_sol, mindist=config.mindist)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


def main():
    """Run forward."""
    msg = 'Running Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))

    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=10, message=msg))
        return

    parallel, run_func, _ = parallel_func(run_forward, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))


if __name__ == '__main__':
    main()
