"""Typing."""

import pathlib
import sys
from typing import Union, List, Dict

import mne

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal  # noqa
else:
    from typing_extensions import TypedDict, Literal  # noqa

PathLike = Union[str, pathlib.Path]


class ArbitraryContrast(TypedDict):
    name: str
    conditions: List[str]
    weights: List[float]


class LogKwargsT(TypedDict):
    msg: str
    extra: Dict[str, str]
    box: str


class ReferenceRunParams(TypedDict):
    montage: mne.channels.DigMontage
    dev_head_t: mne.Transform


try:
    _keys_arbitrary_contrast = set(ArbitraryContrast.__required_keys__)
except Exception:
    _keys_arbitrary_contrast = set(ArbitraryContrast.__annotations__.keys())
