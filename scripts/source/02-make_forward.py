"""
====================
10. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import itertools
import logging

import mne
from mne.utils._bunch import BunchConst
from mne.parallel import parallel_func
from mne_bids import BIDSPath, get_head_mri_trans

import config
from config import (gen_log_message, on_error, failsafe_run, get_runs,
                    get_fs_subject)

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def run_forward(cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    fname_evoked = bids_path.copy().update(suffix='ave')
    fname_trans = bids_path.copy().update(suffix='trans')
    fname_fwd = bids_path.copy().update(suffix='fwd')

    # Generate a head ↔ MRI transformation matrix from the
    # electrophysiological and MRI sidecar files, and save it to an MNE
    # "trans" file in the derivatives folder.
    if cfg.mri_t1_path_generator is None:
        t1_bids_path = None
    else:
        t1_bids_path = BIDSPath(subject=bids_path.subject,
                                session=bids_path.session,
                                root=cfg.bids_root)
        t1_bids_path = cfg.mri_t1_path_generator(t1_bids_path.copy())
        if t1_bids_path.suffix is None:
            t1_bids_path.update(suffix='T1w')
        if t1_bids_path.datatype is None:
            t1_bids_path.update(datatype='anat')

    msg = 'Estimating head ↔ MRI transform'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))

    trans = get_head_mri_trans(
        bids_path.copy().update(run=get_runs(subject=subject)[0],
                                root=cfg.bids_root),
        t1_bids_path=t1_bids_path)
    mne.write_trans(fname_trans, trans)

    fs_subject = get_fs_subject(subject)

    # Create the source space.
    msg = 'Creating source space'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    src = mne.setup_source_space(subject=fs_subject,
                                 subjects_dir=cfg.fs_subjects_dir,
                                 spacing=cfg.spacing,
                                 add_dist=False,
                                 n_jobs=cfg.N_JOBS)

    # Calculate the BEM solution.
    # Here we only use a 3-layers BEM only if EEG is available.
    msg = 'Calculating BEM solution'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))

    if 'eeg' in cfg.ch_types:
        conductivity = (0.3, 0.006, 0.3)
    else:
        conductivity = (0.3,)

    try:
        bem_model = mne.make_bem_model(subject=fs_subject,
                                       subjects_dir=cfg.fs_subjects_dir,
                                       ico=4, conductivity=conductivity)
    except FileNotFoundError:
        message = ("Could not make BEM model due to a missing file. \n"
                   "Can be solved by setting recreate_bem=True in the config "
                   "to force recreation of the BEM model, or by deleting the\n"
                   f" {cfg.bids_roo}/derivatives/freesurfer/"
                   f"subjects/sub-{subject}/bem/ folder")
        raise FileNotFoundError(message)

    bem_sol = mne.make_bem_solution(bem_model)

    # Finally, calculate and save the forward solution.
    msg = 'Calculating forward solution'
    logger.info(gen_log_message(message=msg, step=10, subject=subject,
                                session=session))
    info = mne.io.read_info(fname_evoked)
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem_sol, mindist=cfg.mindist)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


def get_config(subject, session):
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        session=session,
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        mri_t1_path_generator=config.mri_t1_path_generator,
        mindist=config.mindist,
        ch_types=config.ch_types,
        fs_subjects_dir=config.get_fs_subjects_dir(),
        deriv_root=config.get_deriv_root(),
        bids_root=config.get_bids_root()
    )
    return cfg


def main():
    """Run forward."""
    msg = 'Running Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))

    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(gen_log_message(step=10, message=msg))
        return

    parallel, run_func, _ = parallel_func(run_forward, n_jobs=config.N_JOBS)
    parallel(run_func(get_config(subject, session), subject, session)
             for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 10: Create forward solution'
    logger.info(gen_log_message(step=10, message=msg))


if __name__ == '__main__':
    main()
