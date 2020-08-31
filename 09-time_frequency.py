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
import logging

import numpy as np

import mne
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


freqs = np.arange(10, 40)
n_cycles = freqs / 3.


@failsafe_run(on_error=on_error)
def run_time_frequency(subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         modality=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    processing = None
    if config.use_ica or config.use_ssp:
        processing = 'clean'

    fname_in = bids_path.copy().update(suffix='epo', processing=processing,
                                       extension='.fif')

    msg = f'Input: {fname_in}'
    logger.info(gen_log_message(message=msg, step=9, subject=subject,
                                session=session))

    epochs = mne.read_epochs(fname_in)

    for condition in config.time_frequency_conditions:
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True, n_cycles=n_cycles)

        condition_str = condition.replace(op.sep, '').replace('_', '')
        power_fname_out = bids_path.copy().update(
            suffix=f'power+{condition_str}+tfr', extension='.h5')
        itc_fname_out = bids_path.copy().update(
            suffix=f'itc+{condition_str}+tfr', extension='.h5')

        power.save(power_fname_out, overwrite=True)
        itc.save(itc_fname_out, overwrite=True)


def main():
    """Run tf."""
    msg = 'Running Step 9: Time-frequency decomposition'
    logger.info(gen_log_message(message=msg, step=9))

    parallel, run_func, _ = parallel_func(run_time_frequency,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 9: Time-frequency decomposition'
    logger.info(gen_log_message(message=msg, step=9))


if __name__ == '__main__':
    main()
