"""I/O helpers."""

import json_tricks

from .typing import PathLike


def _write_json(fname: PathLike, data: dict) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False)


def _read_json(fname: PathLike) -> dict:
    with open(fname, encoding="utf-8") as f:
        return json_tricks.load(f)
