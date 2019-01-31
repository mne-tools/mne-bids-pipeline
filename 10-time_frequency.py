"""
================================
09. Time-frequency decomposition
================================

The epoched data is transformed to time-frequency domain using morlet wavelets.
Faces and scrambled data sets are used and for both of them, the average power
and inter-trial coherence are computed and saved to disk. Only channel 'EEG070'
is used to save time.
"""

import os.path as op
import numpy as np

import mne
from mne.parallel import parallel_func

from library.config import meg_dir, l_freq, N_JOBS

freqs = np.arange(6, 40)
n_cycles = freqs / 2.


def run_time_frequency(subject_id):
    print("processing subject: %s" % subject_id)
    subject = "sub%03d" % subject_id
    data_path = op.join(meg_dir, subject)
    epochs = mne.read_epochs(op.join(data_path, '%s_highpass-%sHz-epo.fif'
                                     % (subject, l_freq)))

    faces = epochs['face']
    idx = [faces.ch_names.index('EEG065')]
    power_faces, itc_faces = mne.time_frequency.tfr_morlet(
        faces, freqs=freqs, return_itc=True, n_cycles=n_cycles, picks=idx)
    power_scrambled, itc_scrambled = mne.time_frequency.tfr_morlet(
        epochs['scrambled'], freqs=freqs, return_itc=True, n_cycles=n_cycles,
        picks=idx)

    power_faces.save(
        op.join(data_path, '%s_highpass-%sHz-faces-tfr.h5'
                % (subject, l_freq)), overwrite=True)
    itc_faces.save(
        op.join(data_path, '%s_highpass-%sHz-itc_faces-tfr.h5'
                % (subject, l_freq)), overwrite=True)

    power_scrambled.save(
        op.join(data_path, '%s_highpass-%sHz-scrambled-tfr.h5'
                % (subject, l_freq)), overwrite=True)
    itc_scrambled.save(
        op.join(data_path, '%s_highpass-%sHz-itc_scrambled-tfr.h5'
                % (subject, l_freq)), overwrite=True)


parallel, run_func, _ = parallel_func(run_time_frequency, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(1, 20))
