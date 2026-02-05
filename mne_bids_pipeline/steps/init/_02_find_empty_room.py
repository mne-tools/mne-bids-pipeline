"""Find empty-room data matches."""

from types import SimpleNamespace

from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _pl,
    get_datatype,
    get_mf_reference_run_task,
    get_subjects_sessions,
)
from mne_bids_pipeline._import_data import _empty_room_match_path
from mne_bids_pipeline._io import _write_json
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, OutFilesT


def get_input_fnames_find_empty_room(
    *,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    cfg: SimpleNamespace,
) -> InFilesT:
    """Get paths of files required by find_empty_room function."""
    # This must match the logic of _import_data.py
    bids_path_in = BIDSPath(
        subject=subject,
        run=run,
        session=session,
        task=task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        processing=cfg.proc,
        root=cfg.bids_root,
        check=False,
    )
    in_files: InFilesT = dict()
    in_files["raw"] = bids_path_in
    _update_for_splits(in_files, "raw", single=True)
    if hasattr(bids_path_in, "find_matching_sidecar"):
        in_files["sidecar"] = (
            bids_path_in.copy()
            .update(datatype=None, suffix="meg")
            .find_matching_sidecar(extension=".json")
        )
    try:
        fname = bids_path_in.find_empty_room(use_sidecar_only=True)
    except Exception:
        fname = None
    if fname is None and hasattr(bids_path_in, "get_empty_room_candidates"):
        for ci, path in enumerate(bids_path_in.get_empty_room_candidates()):
            in_files[f"empty_room_candidate_{ci}"] = path
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_find_empty_room,
)
def find_empty_room(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str | None,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    raw_path = in_files.pop("raw")
    in_files.pop("sidecar", None)
    try:
        fname = raw_path.find_empty_room(use_sidecar_only=True)
    except (FileNotFoundError, AssertionError, ValueError):
        fname = ""
    if fname is None:
        ending = "empty-room files"
        if len(in_files):  # MNE-BIDS < 0.12 missing get_empty_room_candidates
            ending = f"{len(in_files)} empty-room file{_pl(in_files)}"
        msg = f"Nearest-date matching {ending}"
        logger.info(**gen_log_kwargs(message=msg))
        try:
            fname = raw_path.find_empty_room()
        except (
            ValueError,  # non-MEG data
            AssertionError,  # MNE-BIDS check assert exists()
            FileNotFoundError,
        ):  # MNE-BIDS PR-1080 exists()
            fname = None
        in_files.clear()  # MNE-BIDS find_empty_room should have looked at all
    elif fname == "":
        msg = "Skipping, empty-room match unavailable …"
        logger.info(**gen_log_kwargs(message=msg))
        fname = None  # not downloaded, or EEG data
    elif not fname.fpath.exists():
        msg = f"Path found in sidecar does not exist: {fname.fpath}"
        logger.warning(**gen_log_kwargs(message=msg))
        fname = None  # path found by sidecar but does not exist
    else:
        msg = f"Empty-room match found from sidecar: {fname.fpath.name}"
        logger.info(**gen_log_kwargs(message=msg))
    out_files = dict()
    out_files["empty_room_match"] = _empty_room_match_path(raw_path, cfg)
    _write_json(out_files["empty_room_match"], dict(fname=fname))
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        data_type=get_datatype(config),
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run find_empty_room."""
    if not config.process_empty_room:
        msg = "Skipping, process_empty_room is set to False …"
        logger.info(**gen_log_kwargs(message=msg))
        return
    if get_datatype(config) != "meg":
        msg = "Skipping, empty-room data only relevant for MEG …"
        logger.info(**gen_log_kwargs(message=msg))
        return
    # This will be I/O bound if the sidecar is not complete, so let's not run
    # in parallel.
    logs = list()
    for subject, sessions in get_subjects_sessions(config).items():
        for session in sessions:
            run, task = get_mf_reference_run_task(
                config=config, subject=subject, session=session
            )
            logs.append(
                find_empty_room(
                    cfg=get_config(config=config),
                    exec_params=config.exec_params,
                    subject=subject,
                    session=session,
                    run=run,
                    task=task,
                )
            )
    save_logs(config=config, logs=logs)
