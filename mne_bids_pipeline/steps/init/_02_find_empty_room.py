"""Find empty-room data matches."""

from types import SimpleNamespace
from typing import Dict, Optional

from mne.utils import _pl
from mne_bids import BIDSPath

from ..._config_utils import (
    get_datatype, get_task, get_sessions, get_subjects, get_runs)
from ..._io import _empty_room_match_path, _write_json
from ..._logging import gen_log_kwargs, logger
from ..._run import _update_for_splits, failsafe_run, save_logs


def get_input_fnames_find_empty_room(
    *,
    subject: str,
    session: Optional[str],
    run:  Optional[str],
    cfg: SimpleNamespace
) -> Dict[str, BIDSPath]:
    """Get paths of files required by filter_data function."""
    bids_path_in = BIDSPath(
        subject=subject,
        run=run,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        processing=cfg.proc,
        root=cfg.bids_root,
        check=False
    )
    in_files: Dict[str, BIDSPath] = dict()
    in_files[f'raw_run-{run}'] = bids_path_in
    _update_for_splits(in_files, f'raw_run-{run}', single=True)
    if hasattr(bids_path_in, 'find_matching_sidecar'):
        in_files['sidecar'] = \
            bids_path_in.copy().update(datatype=None).find_matching_sidecar(
                extension='.json')
    try:
        fname = bids_path_in.find_empty_room(use_sidecar_only=True)
    except Exception:
        fname = None
    if fname is None and hasattr(bids_path_in, 'get_empty_room_candidates'):
        for ci, path in enumerate(bids_path_in.get_empty_room_candidates()):
            in_files[f'empty_room_candidate_{ci}'] = path
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_find_empty_room,
)
def find_empty_room(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    run: Optional[str],
    in_files: Dict[str, BIDSPath],
) -> Dict[str, BIDSPath]:
    raw_path = in_files.pop(f'raw_run-{run}')
    in_files.pop('sidecar', None)
    try:
        fname = raw_path.find_empty_room(use_sidecar_only=True)
    except (FileNotFoundError, AssertionError, ValueError):
        fname = ''
    if fname is None:
        # sidecar is very fast and checking all can be slow (seconds), so only
        # log when actually looking through files
        ending = 'empty-room files'
        if len(in_files):  # MNE-BIDS < 0.12 missing get_empty_room_candidates
            ending = f'{len(in_files)} empty-room file{_pl(in_files)}'
        msg = f"Nearest-date matching {ending}"
        logger.info(**gen_log_kwargs(message=msg))
        try:
            fname = raw_path.find_empty_room()
        except (ValueError,  # non-MEG data
                AssertionError,  # MNE-BIDS check assert exists()
                FileNotFoundError):  # MNE-BIDS PR-1080 exists()
            fname = None
        in_files.clear()  # MNE-BIDS find_empty_room should have looked at all
    elif fname == '':
        fname = None  # not downloaded, or EEG data
    elif not fname.fpath.exists():
        fname = None  # path found by sidecar but does not exist
    out_files = dict()
    out_files['empty_room_match'] = _empty_room_match_path(raw_path, cfg)
    _write_json(out_files['empty_room_match'], dict(fname=fname))
    return out_files


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        proc=config.proc,
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.bids_root,
        deriv_root=config.deriv_root,
    )
    return cfg


def main(*, config) -> None:
    """Run find_empty_room."""
    if not config.process_empty_room:
        msg = 'Skipping, process_empty_room is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return
    if get_datatype(config) != 'meg':
        msg = 'Skipping, empty-room data only relevant for MEG …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return
    # This will be I/O bound if the sidecar is not complete, so let's not run
    # in parallel.
    logs = list()
    for subject in get_subjects(config):
        if config.use_maxwell_filter:
            run = config.mf_reference_run
        else:
            run = get_runs(config=config, subject=subject)[0]
        logs.append(find_empty_room(
            cfg=get_config(
                config=config,
            ),
            exec_params=config.exec_params,
            subject=subject,
            session=get_sessions(config)[0],
            run=run,
        ))
    save_logs(config=config, logs=logs)
