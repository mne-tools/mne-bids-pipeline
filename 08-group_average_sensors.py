"""
=====================================
11. Group average at the sensor level
=====================================

The EEG-channel data are averaged for group averages.
"""

import os.path as op

import mne

import config

# Container for all conditions:
all_evokeds = [list() for _ in range(len(config.conditions))]

for subject in config.subjects_list:
    if subject in config.exclude_subjects:
        print("ignoring subject: %s" % subject)
        continue
    else:
        print("processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    fname_ave = op.join(meg_subject_dir, '%s-ave.fif' % subject)

    evokeds = mne.read_evokeds(fname_ave)
    assert len(evokeds) == len(all_evokeds)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container


for idx, evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.combine_evoked(evokeds, 'equal')  # Combine subjects

fname_grand_average = op.join(config.meg_dir, 'grand_average-ave.fif')
print("Saving grand averate: %s" % fname_grand_average)
mne.evoked.write_evokeds(fname_grand_average, all_evokeds)
