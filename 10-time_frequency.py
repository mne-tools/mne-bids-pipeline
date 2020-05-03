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
import itertools

import numpy as np

import mne
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config

freqs = np.arange(10, 40)
n_cycles = freqs / 3.


def run_time_frequency(subject, session=None):
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
        op.join(fpath_deriv, bids_basename + '%s.fif' % extension)

    print("Input: ", fname_in)

    epochs = mne.read_epochs(fname_in)

    for condition in config.time_frequency_conditions:
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True, n_cycles=n_cycles)

        power_fname_out = \
            op.join(fpath_deriv, bids_basename + '_power_%s-tfr.h5'
                    % (condition.replace(op.sep, '')))

        itc_fname_out = \
            op.join(fpath_deriv, bids_basename + '_itc_%s-tfr.h5'
                    % (condition.replace(op.sep, '')))

        power.save(power_fname_out, overwrite=True)
        itc.save(itc_fname_out, overwrite=True)


def main():
    """Run tf."""
    parallel, run_func, _ = parallel_func(run_time_frequency,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))


if __name__ == '__main__':
    main()
