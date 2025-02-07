"""Rejection."""

from collections.abc import Iterable
from typing import Literal

import mne

from ._logging import gen_log_kwargs, logger


def _get_reject(
    *,
    subject: str,
    session: str | None,
    reject: dict[str, float] | Literal["autoreject_global"],
    ch_types: Iterable[Literal["meg", "mag", "grad", "eeg"]],
    param: str,
    epochs: mne.BaseEpochs | None = None,
) -> dict[str, float]:
    if reject is None:
        return dict()

    if reject == "autoreject_global":
        if epochs is None:
            raise ValueError(f'Setting {param}="autoreject_global" is not supported')
        # Automated threshold calculation requested
        import autoreject

        ch_types_autoreject = list(ch_types)
        if "meg" in ch_types_autoreject:
            ch_types_autoreject.remove("meg")
            if "mag" in epochs:
                ch_types_autoreject.append("mag")
            if "grad" in epochs:
                ch_types_autoreject.append("grad")

        msg = "Generating rejection thresholds using autoreject …"
        logger.info(**gen_log_kwargs(message=msg))
        reject_out: dict[str, float] = autoreject.get_rejection_threshold(
            epochs=epochs,
            ch_types=ch_types_autoreject,
            verbose=False,
        )
        return reject_out

    # Only keep thresholds for channel types of interest
    reject = reject.copy()
    ch_types_to_remove: list[str] = list()
    if "meg" not in ch_types:
        ch_types_to_remove.extend(("mag", "grad"))
    if "eeg" not in ch_types:
        ch_types_to_remove.append("eeg")
    for ch_type in ch_types_to_remove:
        try:
            del reject[ch_type]
        except KeyError:
            pass

    return reject
