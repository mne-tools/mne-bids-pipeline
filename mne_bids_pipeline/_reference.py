"""EEG referencing."""

from types import SimpleNamespace

import mne

from ._config_utils import RawEpochsEvokedT
from ._logging import gen_log_kwargs, logger
from ._run import _ignore_warnings


def _set_eeg_reference(
    inst: RawEpochsEvokedT, *, cfg: SimpleNamespace, apply: bool = False
) -> RawEpochsEvokedT:
    """Set the EEG reference, restoring the online reference channel if needed.

    For ``"average"``, the reference is added as a projector. With ``apply=True``
    it is applied and the online reference channel dropped (full-rank data).
    """
    if "eeg" not in cfg.ch_types or "eeg" not in inst:  # e.g., empty-room
        return inst
    if cfg.eeg_reference != "average":
        inst.set_eeg_reference(cfg.eeg_reference, projection=False)
        return inst
    online_ref = cfg.eeg_online_reference_channel
    if not mne._fiff.proj._has_eeg_average_ref_proj(inst.info):
        if online_ref is not None and online_ref not in inst.ch_names:
            msg = f"Adding flat online reference channel {online_ref}"
            logger.info(**gen_log_kwargs(message=msg))
            with _ignore_warnings("Location for this channel is unknown"):
                mne.add_reference_channels(inst, [online_ref], copy=False)
        inst.set_eeg_reference("average", projection=True)
    if apply:
        inst.apply_proj()
        if online_ref in inst.ch_names:
            inst.drop_channels([online_ref])
    return inst
