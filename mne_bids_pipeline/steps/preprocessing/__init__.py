"""Preprocessing."""

from . import _01_data_quality
from . import _02_maxfilter
from . import _03_frequency_filter
from . import _04_make_epochs
from . import _05a_run_ica
from . import _05b_run_ssp
from . import _06a_apply_ica
from . import _06b_apply_ssp
from . import _07_ptp_reject

_STEPS = (
    _01_data_quality,
    _02_maxfilter,
    _03_frequency_filter,
    _04_make_epochs,
    _05a_run_ica,
    _05b_run_ssp,
    _06a_apply_ica,
    _06b_apply_ssp,
    _07_ptp_reject,
)
