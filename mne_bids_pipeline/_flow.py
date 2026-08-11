"""Recording of the files that each pipeline step reads and writes."""

import json
import pathlib
from collections.abc import Mapping

from filelock import FileLock
from mne_bids import BIDSPath

from .typing import TypedDict

_FLOW_DIRNAME = ".pipeline_flow"
_FLOW_VERSION = 1


class FlowEntryT(TypedDict):
    """One recorded call of a pipeline step's worker function."""

    step: str
    func: str
    subject: str | None
    session: str | None
    run: str | None
    task: str | None
    in_files: dict[str, str]
    out_files: dict[str, str]


def _flow_dir(deriv_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(deriv_root) / _FLOW_DIRNAME


def _flow_fname(*, deriv_root: pathlib.Path, subject: str | None) -> pathlib.Path:
    # One file per subject keeps the read-modify-write cheap and keeps parallel
    # subjects off each other's lock.
    stem = "dataset" if subject is None else f"sub-{subject}"
    return _flow_dir(deriv_root) / f"{stem}.json"


def _flow_entry_key(entry: FlowEntryT) -> str:
    keys = ("step", "func", "subject", "session", "run", "task")
    return "|".join(str(entry[key] or "") for key in keys)  # type: ignore[literal-required]


def _read_flow_file(fname: pathlib.Path) -> dict[str, FlowEntryT]:
    if not fname.is_file():
        return dict()
    with FileLock(f"{fname}.lock"):
        try:
            content = json.loads(fname.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            content = dict()
    if content.get("version", None) != _FLOW_VERSION:
        return dict()
    entries: dict[str, FlowEntryT] = content["entries"]
    return entries


def _write_flow_entry(
    *, deriv_root: pathlib.Path, entry: FlowEntryT, only_if_new: bool = False
) -> None:
    """Merge a single recorded step call into the on-disk recording."""
    fname = _flow_fname(deriv_root=deriv_root, subject=entry["subject"])
    fname.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{fname}.lock"):
        entries: dict[str, FlowEntryT] = dict()
        if fname.is_file():
            try:
                content = json.loads(fname.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                content = dict()
            if content.get("version", None) == _FLOW_VERSION:
                entries = content["entries"]
        key = _flow_entry_key(entry)
        if only_if_new and key in entries:
            return
        entries[key] = entry
        fname.write_text(
            json.dumps(dict(version=_FLOW_VERSION, entries=entries), indent=1),
            encoding="utf-8",
        )


def _read_flow_entries(
    *, deriv_root: pathlib.Path, subject: str, session: str | None
) -> list[FlowEntryT]:
    """Get the recorded step calls relevant for one report."""
    fnames = [_flow_fname(deriv_root=deriv_root, subject=None)]
    if subject == "average":
        # Group steps read individual subjects' files, so the group report needs every
        # subject's recording to attribute those inputs to the steps that made them.
        fnames += sorted(_flow_dir(deriv_root).glob("sub-*.json"))
    else:
        fnames.append(_flow_fname(deriv_root=deriv_root, subject=subject))
    entries: list[FlowEntryT] = list()
    for fname in fnames:
        for entry in _read_flow_file(fname).values():
            if entry["session"] in (None, session):
                entries.append(entry)
    entries.sort(key=_flow_entry_key)
    return entries


def _flow_files(files: Mapping[str, object] | None) -> dict[str, str]:
    """Normalize an in_files/out_files mapping to plain path strings."""
    out: dict[str, str] = dict()
    for key, value in (files or dict()).items():
        if key == "__unknown_inputs__":
            continue
        if isinstance(value, tuple):  # out_files carry (path, hash) pairs
            value = value[0]
        if isinstance(value, BIDSPath):
            value = value.fpath
        if isinstance(value, str | pathlib.Path):
            out[key] = str(value)
    return out
