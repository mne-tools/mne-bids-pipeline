from typing import Optional, Literal, Iterable

import mne


def make_epochs(
    *,
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    metadata_tmin: Optional[float] = None,
    metadata_tmax: Optional[float] = None,
    metadata_keep_first: Optional[Iterable[str]] = None,
    metadata_keep_last: Optional[Iterable[str]] = None,
    event_repeated: Literal['error', 'drop', 'merge'],
    decim: int
) -> mne.Epochs:
    """Generate Epochs from raw data.

    No EEG reference will be set and no projectors will be applied. No
    rejection thresholds will be applied. No baseline-correction will be
    performed.
    """
    events, event_id = mne.events_from_annotations(raw)

    # Construct metadata from the epochs
    if metadata_tmin is None:
        metadata_tmin = tmin

    if metadata_tmax is None:
        metadata_tmax = tmax

    metadata, _, _ = mne.epochs.make_metadata(
        events=events, event_id=event_id,
        tmin=metadata_tmin, tmax=metadata_tmax,
        keep_first=metadata_keep_first,
        keep_last=metadata_keep_last,
        sfreq=raw.info['sfreq'])

    # Epoch the data
    # Do not reject based on peak-to-peak or flatness thresholds at this stage
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        proj=False, baseline=None,
                        preload=False, decim=decim,
                        metadata=metadata,
                        event_repeated=event_repeated,
                        reject=None)

    return epochs
