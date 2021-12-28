"""
====================
10. Forward solution
====================

Calculate forward solution for MEG channels.
"""

import itertools
import logging
from typing import Optional

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne.datasets import fetch_fsaverage
from mne_bids import BIDSPath, get_head_mri_trans

import config
from config import gen_log_kwargs, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


def _prepare_forward_fsaverage(cfg):
    assert cfg.fs_subject == 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    bem_sol = cfg.fs_subjects_dir / 'fsaverage' / \
        'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    if not bem_sol.exists():
        fetch_fsaverage(cfg.fs_subjects_dir)
    src = mne.setup_source_space(subject='fsaverage',
                                 subjects_dir=cfg.fs_subjects_dir,
                                 spacing=cfg.spacing,
                                 add_dist=False,
                                 n_jobs=cfg.n_jobs)
    return src, trans, str(bem_sol)


def _prepare_forward(cfg, bids_path, fname_trans):
    # Generate a head ↔ MRI transformation matrix from the
    # electrophysiological and MRI sidecar files, and save it to an MNE
    # "trans" file in the derivatives folder.
    subject, session = bids_path.subject, bids_path.session

    if config.mri_t1_path_generator is None:
        t1_bids_path = None
    else:
        t1_bids_path = BIDSPath(subject=subject,
                                session=session,
                                root=cfg.bids_root)
        t1_bids_path = config.mri_t1_path_generator(t1_bids_path.copy())
        if t1_bids_path.suffix is None:
            t1_bids_path.update(suffix='T1w')
        if t1_bids_path.datatype is None:
            t1_bids_path.update(datatype='anat')

    msg = 'Estimating head ↔ MRI transform'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    trans = get_head_mri_trans(
        bids_path.copy().update(run=cfg.runs[0],
                                root=cfg.bids_root,
                                extension=None),
        t1_bids_path=t1_bids_path,
        fs_subject=cfg.fs_subject,
        fs_subjects_dir=cfg.fs_subjects_dir)

    # Create the source space.
    msg = 'Creating source space'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    src = mne.setup_source_space(subject=cfg.fs_subject,
                                 subjects_dir=cfg.fs_subjects_dir,
                                 spacing=cfg.spacing,
                                 add_dist=False,
                                 n_jobs=cfg.n_jobs)

    # Calculate the BEM solution.
    # Here we only use a 3-layers BEM only if EEG is available.
    msg = 'Calculating BEM solution'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    if 'eeg' in cfg.ch_types:
        conductivity = (0.3, 0.006, 0.3)
    else:
        conductivity = (0.3,)

    try:
        bem_model = mne.make_bem_model(subject=cfg.fs_subject,
                                       subjects_dir=cfg.fs_subjects_dir,
                                       ico=4, conductivity=conductivity)
    except FileNotFoundError:
        message = ("Could not make BEM model due to a missing file. \n"
                   "Can be solved by setting recreate_bem=True in the config "
                   "to force recreation of the BEM model, or by deleting the\n"
                   f" {cfg.bids_root}/derivatives/freesurfer/"
                   f"subjects/sub-{subject}/bem/ folder")
        raise FileNotFoundError(message)

    bem_sol = mne.make_bem_solution(bem_model)
    return src, trans, bem_sol


@failsafe_run(on_error=on_error, script_path=__file__)
def run_forward(*, cfg, subject, session=None):
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

    fname_info = bids_path.copy().update(**cfg.source_info_path_update)
    fname_trans = bids_path.copy().update(suffix='trans')
    fname_fwd = bids_path.copy().update(suffix='fwd')

    if cfg.use_template_mri:
        src, trans, bem_sol = _prepare_forward_fsaverage(cfg)
    else:
        src, trans, bem_sol = _prepare_forward(cfg, bids_path, fname_trans)

    # Finally, calculate and save the forward solution.
    msg = 'Calculating forward solution'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    info = mne.io.read_info(fname_info)
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem_sol, mindist=cfg.mindist)

    mne.write_trans(fname_trans, fwd['mri_head_t'], overwrite=True)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        task=config.get_task(),
        runs=config.get_runs(subject=subject),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        mindist=config.mindist,
        spacing=config.spacing,
        use_template_mri=config.use_template_mri,
        source_info_path_update=config.source_info_path_update,
        ch_types=config.ch_types,
        fs_subject=config.get_fs_subject(subject=subject),
        fs_subjects_dir=config.get_fs_subjects_dir(),
        deriv_root=config.get_deriv_root(),
        bids_root=config.get_bids_root(),
        n_jobs=config.get_n_jobs()
    )
    return cfg


def main():
    """Run forward."""
    if not config.run_source_estimation:
        msg = '    … skipping: run_source_estimation is set to False.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func, _ = parallel_func(run_forward,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(subject=subject), subject=subject,
                 session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
