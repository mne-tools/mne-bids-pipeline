"""
=====================================
11. Group average at the sensor level
=====================================

The M/EEG-channel data are averaged for group averages.
"""

import os.path as op

import mne

import config

# Container for all conditions:
all_evokeds = [list() for _ in range(len(config.conditions))]

for subject in config.subjects_list:
    if subject in config.exclude_subjects:
        print("Ignoring subject: %s" % subject)
        continue
    else:
        print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '-ave'
    fname_in = op.join(meg_subject_dir,
                       config.base_fname.format(**locals()))

    print("Input: ", fname_in)

    evokeds = mne.read_evokeds(fname_in)
    assert len(evokeds) == len(all_evokeds)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container

for idx, evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.combine_evoked(evokeds, 'equal')  # Combine subjects


extension = 'grand_average-ave'
fname_out = op.join(meg_subject_dir,
                    '{0}_{1}.fif'.format(config.study_name, extension))

print("Saving grand averate: %s" % fname_out)
mne.evoked.write_evokeds(fname_out, all_evokeds)


if config.plot:
    ts_args = dict(gfp=True, time_unit='s')
    topomap_args = dict(time_unit='s')

    for idx, evokeds in enumerate(all_evokeds):
        all_evokeds[idx].plot_joint(title=config.conditions[idx],
                                    ts_args=ts_args, topomap_args=topomap_args)
