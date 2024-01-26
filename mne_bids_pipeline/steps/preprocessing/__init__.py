"""Preprocessing."""

from . import (
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
