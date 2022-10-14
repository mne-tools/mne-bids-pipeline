"""I/O helpers."""

import json_tricks


def _write_json(fname, data):
    with open(fname, 'w') as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False)


def _read_json(fname):
    with open(fname, 'r') as f:
        return json_tricks.load(f)


def _empty_room_match_path(run_path, cfg):
    return run_path.copy().update(
        extension='.json', suffix='emptyroommatch', root=cfg.deriv_root)
