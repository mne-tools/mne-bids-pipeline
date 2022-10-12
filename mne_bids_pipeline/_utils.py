import logging
import os
import pathlib
import sys
from typing import Optional, Union, Iterable, List, Dict

import json_tricks
import mne

###############################################################################
# Typing
# ------
if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

PathLike = Union[str, pathlib.Path]


class ArbitraryContrast(TypedDict):
    name: str
    conditions: List[str]
    weights: List[float]


class LogKwargsT(TypedDict):
    msg: str
    extra: Dict[str, str]
    box: str


class ReferenceRunParams(TypedDict):
    montage: mne.channels.DigMontage
    dev_head_t: mne.Transform


###############################################################################
# Logging
# -------

logger = logging.getLogger('mne-bids-pipeline')


def gen_log_kwargs(
    message: str,
    subject: Optional[Union[str, int]] = None,
    session: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    emoji: str = '⏳️',
    script_path: Optional[PathLike] = None,
) -> LogKwargsT:
    if subject is not None:
        subject = f' sub-{subject}'
    if session is not None:
        session = f' ses-{session}'
    if run is not None:
        run = f' run-{run}'

    message = f' {message}'

    script_path = pathlib.Path(os.environ['MNE_BIDS_STUDY_SCRIPT_PATH'])
    step_name = f'{script_path.parent.name}/{script_path.stem}'

    # Choose some to be our standards
    emoji = dict(
        cache='✅',
        skip='⏩',
    ).get(emoji, emoji)
    extra = {
        'step': f'{emoji} {step_name}',
        'box': '│ ',
    }
    if subject:
        extra['subject'] = subject
    if session:
        extra['session'] = session
    if run:
        extra['run'] = run

    kwargs: LogKwargsT = {
        'msg': message,
        'extra': extra,
    }
    return kwargs


###############################################################################
# Rejection
# ---------

def _get_reject(
    *,
    subject: str,
    session: str,
    reject: Union[Dict[str, float], Literal['autoreject_global']],
    ch_types: Iterable[Literal['meg', 'mag', 'grad', 'eeg']],
    epochs: Union[mne.BaseEpochs, None],
    decim: int,
) -> Dict[str, float]:
    if reject is None:
        return dict()

    if reject == 'autoreject_global':
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
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))
        reject = autoreject.get_rejection_threshold(
            epochs=epochs, ch_types=ch_types_autoreject,
            decim=decim, verbose=False,
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


###############################################################################
# I/O helpers
# -----------

def _write_json(fname, data):
    with open(fname, 'w') as f:
        json_tricks.dump(data, fp=f, allow_nan=True, sort_keys=False)


def _read_json(fname):
    with open(fname, 'r') as f:
        data = json_tricks.load(f)
    return data
