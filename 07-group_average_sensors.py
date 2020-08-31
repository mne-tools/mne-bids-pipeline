"""
=====================================
11. Group average at the sensor level
=====================================

The M/EEG-channel data are averaged for group averages.
"""

import os
import os.path as op
from collections import defaultdict
import logging

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_message

logger = logging.getLogger('mne-study-template')


msg = 'Running Step 7: Grand-average sensor data'
logger.info(gen_log_message(step=7, message=msg))

# Container for all conditions:
all_evokeds = defaultdict(list)

# XXX to fix
if config.get_sessions():
    session = config.get_sessions()[0]
else:
    session = None

for subject in config.get_subjects():
    fname_in = BIDSPath(subject=subject,
                        session=session,
                        task=config.get_task(),
                        acquisition=config.acq,
                        run=None,
                        recording=config.rec,
                        space=config.space,
                        suffix='ave',
                        extension='.fif',
                        modality=config.get_datatype(),
                        root=config.deriv_root,
                        check=False)

    msg = f'Input: {fname_in}'
    logger.info(gen_log_message(message=msg, step=7, subject=subject,
                                session=session))

    evokeds = mne.read_evokeds(fname_in)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container

for idx, evokeds in all_evokeds.items():
    all_evokeds[idx] = mne.grand_average(
        evokeds, interpolate_bads=config.interpolate_bads_grand_average
    )  # Combine subjects

subject = 'average'

fname_out = BIDSPath(subject=subject,
                     session=session,
                     task=config.get_task(),
                     acquisition=config.acq,
                     run=None,
                     processing=config.proc,
                     recording=config.rec,
                     space=config.space,
                     suffix='ave',
                     extension='.fif',
                     modality=config.get_datatype(),
                     root=config.deriv_root,
                     check=False)

if not fname_out.fpath.parent.exists():
    os.makedirs(fname_out.fpath.parent)

msg = f'Saving grand-averaged sensor data: {fname_out}'
logger.info(gen_log_message(message=msg, step=7, subject=subject,
                            session=session))
mne.evoked.write_evokeds(fname_out, list(all_evokeds.values()))


def main():
    """Plot evokeds."""
    if not config.interactive:
        return

    for evoked in all_evokeds.values():
        evoked.plot()

    # ts_args = dict(gfp=True, time_unit='s')
    # topomap_args = dict(time_unit='s')

    # for idx, evokeds in enumerate(all_evokeds):
    #     all_evokeds[idx].plot_joint(title=config.conditions[idx],
    #                                 ts_args=ts_args, topomap_args=topomap_args)  # noqa: E501


if __name__ == '__main__':
    main()


msg = 'Completed Step 7: Grand-average sensor data'
logger.info(gen_log_message(step=7, message=msg))
