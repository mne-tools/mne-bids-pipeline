"""I/O helpers."""

from typing import Any

import json_tricks
from mne_bids import BIDSPath

from .typing import PathLike


def _write_json(
    fname: PathLike | BIDSPath,
    data: dict[str, Any] | None,
    *,
    indent: int | None = None,
) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False, indent=indent)


def _read_json(fname: PathLike | BIDSPath) -> Any:
    with open(fname, encoding="utf-8") as f:
        return json_tricks.load(f)
