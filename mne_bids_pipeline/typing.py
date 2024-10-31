"""Custom data types for MNE-BIDS-Pipeline."""

import pathlib
import sys
from typing import Annotated, Any, Literal, TypeAlias

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import mne
import numpy as np
from mne_bids import BIDSPath
from numpy.typing import ArrayLike
from pydantic import PlainValidator

PathLike = str | pathlib.Path

__all__ = [
    "ArbitraryContrast",
    "DigMontageType",
    "FloatArrayLike",
    "FloatArrayT",
    "LogKwargsT",
    "OutFilesT",
    "PathLike",
    "RunTypeT",
    "RunKindT",
    "TypedDict",
]


ShapeT: TypeAlias = tuple[int, ...] | tuple[int]
IntArrayT: TypeAlias = np.ndarray[ShapeT, np.dtype[np.integer[Any]]]
FloatArrayT: TypeAlias = np.ndarray[ShapeT, np.dtype[np.floating[Any]]]
OutFilesT: TypeAlias = dict[str, tuple[str, str | float]]
InFilesT: TypeAlias = dict[str, BIDSPath]  # Only BIDSPath
InFilesPathT: TypeAlias = dict[str, BIDSPath | pathlib.Path]  # allow generic Path too


class ArbitraryContrast(TypedDict):
    """Statistical contrast with arbitrary weights."""

    name: str
    conditions: list[str]
    weights: list[float]


class LogKwargsT(TypedDict):
    """Container for logger keyword arguments."""

    msg: str
    extra: dict[str, str]


RunTypeT = Literal["experimental", "empty-room", "resting-state"]
RunKindT = Literal["orig", "sss", "filt"]


def assert_float_array_like(val: Any) -> FloatArrayT:
    """Convert the input into a NumPy float array."""
    # https://docs.pydantic.dev/latest/errors/errors/#custom-errors
    # Should raise ValueError or AssertionError... NumPy should do this for us
    return np.array(val, dtype=np.float64)


FloatArrayLike = Annotated[
    ArrayLike,
    # PlainValidator will skip internal validation attempts for ArrayLike
    PlainValidator(assert_float_array_like),
]


def assert_dig_montage(val: mne.channels.DigMontage) -> mne.channels.DigMontage:
    """Assert that the input is a DigMontage."""
    assert isinstance(val, mne.channels.DigMontage)
    return val


DigMontageType = Annotated[
    mne.channels.DigMontage,
    PlainValidator(assert_dig_montage),
]
