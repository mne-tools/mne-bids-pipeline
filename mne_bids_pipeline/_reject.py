"""Rejection."""

from typing import Optional, Union, Iterable, Dict, Literal

import mne

from ._logging import logger, gen_log_kwargs


def _get_reject(
    *,
    subject: str,
    session: Optional[str],
    reject: Union[Dict[str, float], Literal['autoreject_global']],
    ch_types: Iterable[Literal['meg', 'mag', 'grad', 'eeg']],
    param: str,
    epochs: Optional[mne.BaseEpochs] = None,
) -> Dict[str, float]:
    if reject is None:
        return dict()

    if reject == 'autoreject_global':
        if epochs is None:
            raise ValueError(
                f'Setting {param}="autoreject_global" is not supported')
        # Automated threshold calculation requested
        import autoreject

        ch_types_autoreject = list(ch_types)
        if 'meg' in ch_types_autoreject:
            ch_types_autoreject.remove('meg')
            if 'mag' in epochs:
                ch_types_autoreject.append('mag')
            if 'grad' in epochs:
                ch_types_autoreject.append('grad')

        msg = 'Generating rejection thresholds using autoreject …'
        logger.info(**gen_log_kwargs(message=msg))
        reject = autoreject.get_rejection_threshold(
            epochs=epochs,
            ch_types=ch_types_autoreject,
            verbose=False,
        )
        return reject

    # Only keep thresholds for channel types of interest
    reject = reject.copy()
    if ch_types == ['eeg']:
        ch_types_to_remove = ('mag', 'grad')
    else:
        ch_types_to_remove = ('eeg',)

    for ch_type in ch_types_to_remove:
        try:
            del reject[ch_type]
        except KeyError:
            pass

    return reject
