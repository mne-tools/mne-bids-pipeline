"""Typing."""

import pathlib
from typing import Union, List, Dict, TypedDict

import mne

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
