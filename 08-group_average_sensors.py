"""
=====================================
11. Group average at the sensor level
=====================================

The EEG-channel data are averaged for group averages.
"""

import os.path as op

import mne

from library.config import meg_dir, l_freq, exclude_subjects

all_evokeds = [list() for _ in range(7)]  # Container for all the categories

for run in range(1, 20):
    if run in exclude_subjects:
        continue
    subject = "sub%03d" % run
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)

    evokeds = mne.read_evokeds(
        op.join(meg_dir, subject, '%s_highpass-%sHz-ave.fif'
                % (subject, l_freq)))
    assert len(evokeds) == len(all_evokeds)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container


for idx, evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.combine_evoked(evokeds, 'equal')  # Combine subjects

mne.evoked.write_evokeds(
    op.join(meg_dir, 'grand_average_highpass-%sHz-ave.fif' % l_freq),
    all_evokeds)
