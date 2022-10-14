"""I/O helpers."""

import json_tricks


def _write_json(fname, data):
    with open(fname, 'w') as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False)
