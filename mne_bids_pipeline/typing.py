"""Typing."""

import pathlib
import sys
from typing import Union, List, Dict
from typing_extensions import Annotated

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
from pydantic import PlainValidator
import mne

PathLike = Union[str, pathlib.Path]


class ArbitraryContrast(TypedDict):
    name: str
    conditions: List[str]
    weights: List[float]


class LogKwargsT(TypedDict):
    msg: str
    extra: Dict[str, str]


class ReferenceRunParams(TypedDict):
    montage: mne.channels.DigMontage
    dev_head_t: mne.Transform


def assert_float_array_like(val):
    # https://docs.pydantic.dev/latest/errors/errors/#custom-errors
    # Should raise ValueError or AssertionError... NumPy should do this for us
    return np.array(val, dtype="float")


FloatArrayLike = Annotated[
    ArrayLike,
    # PlainValidator will skip internal validation attempts for ArrayLike
    PlainValidator(assert_float_array_like),
]


def assert_dig_montage(val):
    assert isinstance(val, mne.channels.DigMontage)
    return val


DigMontageType = Annotated[
    mne.channels.DigMontage,
    PlainValidator(assert_dig_montage),
]
