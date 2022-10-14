"""Find empty-room data matches."""

from types import SimpleNamespace

from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, failsafe_run, _update_for_splits

from ..._io import _write_json, _empty_room_match_path
from ..._logging import logger


def get_input_fnames_find_empty_room(*, subject, session, run, cfg):
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
    in_files = dict()
    in_files[f'raw_run-{run}'] = bids_path_in
    _update_for_splits(in_files, f'raw_run-{run}', single=True)
    try:
        fname = in_files[f'raw_run-{run}'].find_empty_room(
            use_sidecar_only=True)
    except (FileNotFoundError, AssertionError, ValueError):
        return in_files
    if fname is not None:
        # TODO: We should include the sidecar here...
        pass
    else:
        # TODO: Add all empty-room files that will be traversed by MNE-BIDS.
        # For this, we need to refactor MNE-BIDS to get a list of possible
        # empty-room matches.
        pass
    return in_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_find_empty_room)
def find_empty_room(*, subject, session, run, in_files, cfg):
    raw_path = in_files.pop(f'raw_run-{run}')
    try:
        fname = raw_path.find_empty_room(use_sidecar_only=True)
    except (FileNotFoundError, AssertionError, ValueError):
        fname = ''
    if fname is None:
        # sidecar is very fast and checking all can be slow (seconds), so only
        # log when actually looking through files
        msg = (
            f"Nearest-date matching {len(in_files)} empty-room files")
        logger.info(**gen_log_kwargs(
            message=msg, subject=subject, session=session, run=run))
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
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        proc=config.proc,
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.get_bids_root(),
        deriv_root=config.deriv_root,
    )
    return cfg


def main():
    """Run find_empty_room."""
    if not config.process_empty_room:
        msg = 'Skipping, process_empty_room is set to False …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return
    if config.get_datatype() != 'meg':
        msg = 'Skipping, empty-room data only relevant for MEG …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return
    # This will be I/O bound if the sidecar is not complete, so let's not run
    # in parallel.
    logs = list()
    for subject in config.get_subjects():
        if config.use_maxwell_filter:
            run = config.mf_reference_run
        else:
            run = config.get_runs(subject=subject)[0]
        logs.append(find_empty_room(
            cfg=get_config(),
            subject=subject,
            session=config.get_sessions()[0],
            run=run,
        ))
    config.save_logs(logs)


if __name__ == '__main__':
    main()
