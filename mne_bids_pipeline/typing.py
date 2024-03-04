"""Custom data types for MNE-BIDS-Pipeline."""

import pathlib
import sys
from typing import Annotated, Union

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import mne
import numpy as np
from numpy.typing import ArrayLike
from pydantic import PlainValidator

PathLike = Union[str, pathlib.Path]


class ArbitraryContrast(TypedDict):
    """Statistical contrast with arbitrary weights."""

    name: str
    conditions: list[str]
    weights: list[float]


class LogKwargsT(TypedDict):
    """Container for logger keyword arguments."""

    msg: str
    extra: dict[str, str]


def assert_float_array_like(val):
    """Convert the input into a NumPy float array."""
    # https://docs.pydantic.dev/latest/errors/errors/#custom-errors
    # Should raise ValueError or AssertionError... NumPy should do this for us
    return np.array(val, dtype="float")


FloatArrayLike = Annotated[
    ArrayLike,
    # PlainValidator will skip internal validation attempts for ArrayLike
    PlainValidator(assert_float_array_like),
]


def assert_dig_montage(val):
    """Assert that the input is a DigMontage."""
    assert isinstance(val, mne.channels.DigMontage)
    return val


DigMontageType = Annotated[
    mne.channels.DigMontage,
    PlainValidator(assert_dig_montage),
]
