"""Source-space analysis."""

from . import (
    _01_make_bem_surfaces,
    _02_make_bem_solution,
    _03_setup_source_space,
    _04_make_forward,
    _05_make_inverse,
    _99_group_average,
)

_STEPS = (
    _01_make_bem_surfaces,
    _02_make_bem_solution,
    _03_setup_source_space,
    _04_make_forward,
    _05_make_inverse,
    _99_group_average,
)
