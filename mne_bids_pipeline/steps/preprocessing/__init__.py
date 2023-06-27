"""Preprocessing."""

from . import _01_data_quality
from . import _02_head_pos
from . import _03_maxfilter
from . import _04_frequency_filter
from . import _05_make_epochs
from . import _06a_run_ica
from . import _06b_run_ssp
from . import _07a_apply_ica
from . import _07b_apply_ssp
from . import _08_ptp_reject

_STEPS = (
    _01_data_quality,
    _02_head_pos,
    _03_maxfilter,
    _04_frequency_filter,
    _05_make_epochs,
    _06a_run_ica,
    _06b_run_ssp,
    _07a_apply_ica,
    _07b_apply_ssp,
    _08_ptp_reject,
)
