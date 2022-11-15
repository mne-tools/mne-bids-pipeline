"""Filesystem initialization and dataset inspection."""

from . import _00_init_derivatives_dir
from . import _01_find_empty_room

SCRIPTS = (
    _00_init_derivatives_dir,
    _01_find_empty_room,
)
