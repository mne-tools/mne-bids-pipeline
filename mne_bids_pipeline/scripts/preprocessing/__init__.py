"""Preprocessing."""

from . import _01_maxfilter
from . import _02_frequency_filter
from . import _03_make_epochs
from . import _04a_run_ica
from . import _04b_run_ssp
from . import _05a_apply_ica
from . import _05b_apply_ssp
from . import _06_ptp_reject

SCRIPTS = (
    _01_maxfilter,
    _02_frequency_filter,
    _03_make_epochs,
    _04a_run_ica,
    _04b_run_ssp,
    _05a_apply_ica,
    _05b_apply_ssp,
    _06_ptp_reject,
)
