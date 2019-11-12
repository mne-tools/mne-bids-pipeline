"""
=====================================
11. Group average at the sensor level
=====================================

The M/EEG-channel data are averaged for group averages.
"""

import os.path as op
from collections import defaultdict

import mne
from mne_bids import make_bids_basename

import config

# Container for all conditions:
all_evokeds = defaultdict(list)

# XXX to fix
session = config.sessions[0]

for subject in config.subjects_list:
    if subject in config.exclude_subjects:
        print("Ignoring subject: %s" % subject)
        continue
    else:
        print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    if config.use_ica or config.use_ssp:
        extension = '_cleaned-epo'
    else:
        extension = '-epo'

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)

    fname_in = \
        op.join(fpath_deriv, bids_basename + '-ave.fif')

    print("Input: ", fname_in)

    evokeds = mne.read_evokeds(fname_in)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container

for idx, evokeds in all_evokeds.items():
    all_evokeds[idx] = mne.combine_evoked(evokeds, 'equal')  # Combine subjects


extension = 'grand_average-ave'
fname_out = op.join(config.bids_root, 'derivatives', config.PIPELINE_NAME,
                    '{0}_{1}.fif'.format(config.task, extension))

print("Saving grand averate: %s" % fname_out)
mne.evoked.write_evokeds(fname_out, list(all_evokeds.values()))


def main():
    """Plot evokeds."""
    if not config.plot:
        return
    
    ts_args = dict(gfp=True, time_unit='s')
    topomap_args = dict(time_unit='s')
    
    for idx, evokeds in enumerate(all_evokeds):
        all_evokeds[idx].plot_joint(title=config.conditions[idx],
                                    ts_args=ts_args, topomap_args=topomap_args)  # noqa: E501
        # XXX error in plotting, and plots were generated twice
   
if __name__ == '__main__':
    main()
