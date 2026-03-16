"""Setup source space.

Set up source space for forward and inverse computation.
"""

from types import SimpleNamespace

import mne

from mne_bids_pipeline._config_utils import (
    get_fs_subject,
    get_fs_subjects_dir,
    get_sessions,
    get_subjects_sessions,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._run import _prep_out_files_path, failsafe_run, save_logs
from mne_bids_pipeline.typing import InFilesPathT, OutFilesT


def get_input_fnames_setup_source_space(
    *, cfg: SimpleNamespace, subject: str
) -> InFilesPathT:
    in_files = dict()
    surf_path = cfg.fs_subjects_dir / cfg.fs_subject / "surf"
    for hemi in ("lh", "rh"):
        for kind in ("sphere", "sphere.reg", "white"):
            in_files["surf-{hemi}-{kind}"] = surf_path / f"{hemi}.{kind}"
    return in_files


def get_output_fnames_setup_source_space(
    *, cfg: SimpleNamespace, subject: str
) -> InFilesPathT:
    out_files = dict()
    out_files["src"] = (
        cfg.fs_subjects_dir
        / cfg.fs_subject
        / "bem"
        / f"{cfg.fs_subject}-{cfg.spacing}-src.fif"
    )
    return out_files


@failsafe_run(
    get_input_fnames=get_input_fnames_setup_source_space,
    get_output_fnames=get_output_fnames_setup_source_space,
)
def run_setup_source_space(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    in_files: InFilesPathT,
) -> OutFilesT:
    msg = f"Creating source space with spacing {repr(cfg.spacing)}"
    logger.info(**gen_log_kwargs(message=msg, subject=subject))
    src = mne.setup_source_space(
        cfg.fs_subject,
        spacing=cfg.spacing,
        subjects_dir=cfg.fs_subjects_dir,
        add_dist="patch",
    )
    in_files.clear()  # all used by setup_source_space
    out_files = get_output_fnames_setup_source_space(cfg=cfg, subject=subject)
    mne.write_source_spaces(out_files["src"], src, overwrite=True)
    return _prep_out_files_path(
        exec_params=exec_params,
        out_files=out_files,
        check_relative=cfg.fs_subjects_dir,
    )


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        spacing=config.spacing,
        use_template_mri=config.use_template_mri,
        fs_subject=get_fs_subject(config=config, subject=subject, session=session),
        fs_subjects_dir=get_fs_subjects_dir(config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run forward."""
    if not config.run_source_estimation:
        msg = "Skipping, run_source_estimation is set to False …"
        logger.info(**gen_log_kwargs(message=msg))
        return

    if config.use_template_mri is not None:
        sub_ses = {config.use_template_mri: get_sessions(config=config)}
    else:
        sub_ses = get_subjects_sessions(config=config)

    ss = [
        (subject, session)
        for subject, sessions in sub_ses.items()
        for session in sessions
    ]
    del sub_ses
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_setup_source_space, exec_params=config.exec_params, n_iter=len(ss)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session,
                ),
                exec_params=config.exec_params,
                subject=subject,
            )
            for subject, session in ss
        )
    save_logs(config=config, logs=logs)
