"""I/O helpers."""

from typing import Any

from mne_bids import BIDSPath

from .typing import PathLike

# Every FileLock gets this: long enough for a big report write on a slow (e.g.
# networked) filesystem, finite so that a lock we will never get raises Timeout
# naming the file rather than hanging the run forever
_LOCK_TIMEOUT = 600.0


def _write_json(
    fname: PathLike | BIDSPath,
    data: dict[str, Any] | None,
    *,
    indent: int | None = None,
) -> None:
    import json_tricks

    with open(fname, "w", encoding="utf-8") as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False, indent=indent)


def _read_json(fname: PathLike | BIDSPath) -> Any:
    import json_tricks

    with open(fname, encoding="utf-8") as f:
        return json_tricks.load(f)
