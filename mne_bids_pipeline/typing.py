"""Custom data types for MNE-BIDS-Pipeline."""

import pathlib
import sys
from collections.abc import Hashable, Sequence
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import mne
import numpy as np
from mne_bids import BIDSPath
from numpy.typing import ArrayLike
from pydantic import AfterValidator, Field, PlainValidator
from pydantic_core import PydanticCustomError

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
BaselineTypeT: TypeAlias = tuple[float | None, float | None] | None
RunsTypeT: TypeAlias = Sequence[str] | Literal["all"]
ConditionsTypeT: TypeAlias = Sequence[str] | dict[str, str]


class ArbitraryContrast(TypedDict):
    """Statistical contrast with arbitrary weights."""

    name: str
    conditions: list[str]
    weights: list[float]


ContrastSequenceT: TypeAlias = Sequence[tuple[str, str] | ArbitraryContrast]


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

T = TypeVar("T", bound=Hashable)


def _validate_unique_sequence(v: Sequence[T]) -> Sequence[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_sequence", "Sequence items must be unique")
    return v


UniqueSequence = Annotated[
    Sequence[T],
    AfterValidator(_validate_unique_sequence),
    Field(json_schema_extra={"uniqueItems": True}),
]
