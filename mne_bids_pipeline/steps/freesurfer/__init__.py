"""FreeSurfer-related processing.

These steps are not run by default.
"""

from . import _01_recon_all
from . import _02_coreg_surfaces

_STEPS = (
    _01_recon_all,
    _02_coreg_surfaces
)
