"""I/O helpers."""

from types import SimpleNamespace

import json_tricks
from mne_bids import BIDSPath

from .typing import PathLike


def _write_json(fname: PathLike, data: dict) -> None:
    with open(fname, 'w', encoding='utf-8') as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False)


def _read_json(fname: PathLike) -> dict:
    with open(fname, 'r', encoding='utf-8') as f:
        return json_tricks.load(f)


def _empty_room_match_path(
    run_path: BIDSPath,
    cfg: SimpleNamespace
) -> BIDSPath:
    return run_path.copy().update(
        extension='.json', suffix='emptyroommatch', root=cfg.deriv_root
    )
