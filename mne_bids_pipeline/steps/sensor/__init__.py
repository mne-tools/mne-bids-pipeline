"""Sensor-space analysis."""

from . import _01_make_evoked
from . import _02_decoding_full_epochs
from . import _03_decoding_time_by_time
from . import _04_time_frequency
from . import _05_decoding_csp
from . import _06_make_cov
from . import _99_group_average

_STEPS = (
    _01_make_evoked,
    _02_decoding_full_epochs,
    _03_decoding_time_by_time,
    _04_time_frequency,
    _05_decoding_csp,
    _06_make_cov,
    _99_group_average,
)
