"""Forward solution.

Calculate forward solution for M/EEG channels.
"""

from types import SimpleNamespace
from typing import Optional

import mne
import numpy as np
from mne.coreg import Coregistration
from mne_bids import BIDSPath, get_head_mri_trans

from ..._config_utils import (
    _bids_kwargs,
    _get_bem_conductivity,
    _meg_in_ch_types,
    get_fs_subject,
    get_fs_subjects_dir,
    get_runs,
    get_sessions,
    get_subjects,
)
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._report import _open_report, _render_bem
from ..._run import _prep_out_files, failsafe_run, save_logs


def _prepare_trans_template(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    info: mne.Info,
) -> mne.transforms.Transform:
    assert isinstance(cfg.use_template_mri, str)
    assert cfg.use_template_mri == cfg.fs_subject

    if cfg.fs_subject != "fsaverage" and not cfg.adjust_coreg:
        raise ValueError(
            "Adjusting the coregistration is mandatory "
            "when using a template MRI different from "
            "fsaverage"
        )
    if cfg.fs_subject == "fsaverage" and not cfg.adjust_coreg:
        trans = (
            cfg.fs_subjects_dir / cfg.fs_subject / "bem" / f"{cfg.fs_subject}-trans.fif"
        )
    else:
        fiducials = "estimated"  # get fiducials from fsaverage
        logger.info(**gen_log_kwargs("Matching template MRI using fiducials"))
        coreg = Coregistration(
            info, cfg.fs_subject, cfg.fs_subjects_dir, fiducials=fiducials
        )
        # Adapted from MNE-Python
        coreg.fit_fiducials(verbose=False)
        dist = np.median(coreg.compute_dig_mri_distances() * 1000)
        logger.info(**gen_log_kwargs(f"Median dig ↔ MRI distance: {dist:6.2f} mm"))
        trans = coreg.trans

    return trans


def _prepare_trans_subject(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    bids_path: BIDSPath,
) -> mne.transforms.Transform:
    # Generate a head ↔ MRI transformation matrix from the
    # electrophysiological and MRI sidecar files, and save it to an MNE
    # "trans" file in the derivatives folder.

    msg = "Computing head ↔ MRI transform from matched fiducials"
    logger.info(**gen_log_kwargs(message=msg))

    trans = get_head_mri_trans(
        bids_path.copy().update(run=cfg.runs[0], root=cfg.bids_root, extension=None),
        t1_bids_path=cfg.t1_bids_path,
        fs_subject=cfg.fs_subject,
        fs_subjects_dir=cfg.fs_subjects_dir,
        kind=cfg.landmarks_kind,
    )

    return trans


def get_input_fnames_forward(*, cfg, subject, session):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    in_files = dict()
    in_files["info"] = bids_path.copy().update(**cfg.source_info_path_update)
    bem_path = cfg.fs_subjects_dir / cfg.fs_subject / "bem"
    _, tag = _get_bem_conductivity(cfg)
    in_files["bem"] = bem_path / f"{cfg.fs_subject}-{tag}-bem-sol.fif"
    in_files["src"] = bem_path / f"{cfg.fs_subject}-{cfg.spacing}-src.fif"
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_forward,
)
def run_forward(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )

    # Info
    info = mne.io.read_info(in_files.pop("info"))
    info = mne.pick_info(
        info,
        mne.pick_types(
            info,
            meg=_meg_in_ch_types(cfg.ch_types),
            eeg="eeg" in cfg.ch_types,
            exclude=[],
        ),
    )

    # BEM
    bem = in_files.pop("bem")

    # source space
    src = in_files.pop("src")

    # trans
    if cfg.use_template_mri is not None:
        trans = _prepare_trans_template(
            cfg=cfg,
            subject=subject,
            session=session,
            info=info,
        )
    else:
        trans = _prepare_trans_subject(
            cfg=cfg,
            subject=subject,
            session=session,
            exec_params=exec_params,
            bids_path=bids_path,
        )

    msg = "Calculating forward solution"
    logger.info(**gen_log_kwargs(message=msg))
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem, mindist=cfg.mindist
    )
    out_files = dict()
    out_files["trans"] = bids_path.copy().update(suffix="trans")
    out_files["forward"] = bids_path.copy().update(suffix="fwd")
    mne.write_trans(out_files["trans"], fwd["mri_head_t"], overwrite=True)
    mne.write_forward_solution(out_files["forward"], fwd, overwrite=True)

    # Report
    with _open_report(
        cfg=cfg, exec_params=exec_params, subject=subject, session=session
    ) as report:
        msg = "Adding forward information to report"
        logger.info(**gen_log_kwargs(message=msg))
        _render_bem(report=report, cfg=cfg, subject=subject, session=session)
        msg = "Rendering sensor alignment (coregistration)"
        logger.info(**gen_log_kwargs(message=msg))
        report.add_trans(
            trans=trans,
            info=info,
            title="Sensor alignment",
            subject=cfg.fs_subject,
            subjects_dir=cfg.fs_subjects_dir,
            alpha=1,
            replace=True,
        )
        msg = "Rendering forward solution"
        logger.info(**gen_log_kwargs(message=msg))
        report.add_forward(
            forward=fwd,
            title="Forward solution",
            subject=cfg.fs_subject,
            subjects_dir=cfg.fs_subjects_dir,
            replace=True,
        )

    assert len(in_files) == 0, in_files
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> SimpleNamespace:
    if config.mri_t1_path_generator is None:
        t1_bids_path = None
    else:
        t1_bids_path = BIDSPath(subject=subject, session=session, root=config.bids_root)
        t1_bids_path = config.mri_t1_path_generator(t1_bids_path.copy())
        if t1_bids_path.suffix is None:
            t1_bids_path.update(suffix="T1w")
        if t1_bids_path.datatype is None:
            t1_bids_path.update(datatype="anat")
    if config.mri_landmarks_kind is None:
        landmarks_kind = None
    else:
        landmarks_kind = config.mri_landmarks_kind(
            BIDSPath(subject=subject, session=session)
        )

    cfg = SimpleNamespace(
        runs=get_runs(config=config, subject=subject),
        mindist=config.mindist,
        spacing=config.spacing,
        use_template_mri=config.use_template_mri,
        adjust_coreg=config.adjust_coreg,
        source_info_path_update=config.source_info_path_update,
        ch_types=config.ch_types,
        fs_subject=get_fs_subject(config=config, subject=subject),
        fs_subjects_dir=get_fs_subjects_dir(config),
        t1_bids_path=t1_bids_path,
        landmarks_kind=landmarks_kind,
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run forward."""
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(run_forward, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject, session=session),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
