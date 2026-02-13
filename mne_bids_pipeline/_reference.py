"""EEG reference utilities."""

from ._logging import gen_log_kwargs, logger
from types import SimpleNamespace
import mne

def set_initial_average_reference(inst, cfg: SimpleNamespace):
    """Set an average EEG reference with the option to add the online flat channel before re-referencing.

    If `cfg.add_online_reference_channel` is True and the specified `cfg.eeg_online_reference_channel` is not yet present,
    it is added as a flat reference channel, increasing the number of channels in `inst` by one.

    Note: 
    - The average reference is added as a projection and not yet applied.
    - If you want to drop the online reference channel after re-referencing,
      first apply the projection and then drop the channel.
    """

    assert_msg = "An average reference projection has already been applied to the data. You cannot add the online reference as a flat channel anymore. Given this function is rather used internally, you might want to raise an issue on GitHub."
    assert not mne._fiff.proj._has_eeg_average_ref_proj(inst.info), assert_msg

    if cfg.add_online_reference_channel:
        assert cfg.eeg_online_reference_channel is not None, "To add the online reference channel as flat channel before re-referencing, `eeg_online_reference_channel` must be provided."

        if cfg.eeg_online_reference_channel in inst.ch_names:
            msg = f"Specified online reference channel {cfg.eeg_online_reference_channel} exists already. Double-check if this is really the reference channel name, if it is, consider setting `add_online_reference_channel` to `False` in case it is indeed a flat channel."
            logger.warning(**gen_log_kwargs(message=msg))
        else:
            msg = f"Online reference channel {cfg.eeg_online_reference_channel} will be added as flat channel."
            logger.info(**gen_log_kwargs(message=msg))

            mne.add_reference_channels(inst, ref_channels=[cfg.eeg_online_reference_channel], copy=False)

    else:
        msg = "Re-referencing to average reference assuming the online reference channel was already added as a flat channel."
        logger.info(**gen_log_kwargs(message=msg))

    # We use this instead of projection=False to later being able to check if the average projection was already applied
    inst.set_eeg_reference("average", projection=True)#.apply_proj()
    
    return inst