"""
================================
08. Time-frequency decomposition
================================

The epoched data is transformed to time-frequency domain using morlet wavelets.
The average power and inter-trial coherence are computed and saved to disk.
"""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import numpy as np

import mne

from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, failsafe_run, sanitize_cond_name
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(script_path=__file__)
def run_time_frequency(*, cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    processing = None
    if cfg.spatial_filter is not None:
        processing = 'clean'

    fname_in = bids_path.copy().update(suffix='epo', processing=processing,
                                       extension='.fif')

    msg = f'Input: {fname_in.basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(fname_in)
    if cfg.analyze_channels:
        # We special-case the average reference here.
        # See time-by-time decoding script for more info.
        if 'eeg' in cfg.ch_types and cfg.eeg_reference == 'average':
            epochs.set_eeg_reference('average')
        else:
            epochs.apply_proj()
        epochs.pick(cfg.analyze_channels)

    if cfg.time_frequency_subtract_evoked:
        epochs.subtract_evoked()

    freqs = np.arange(cfg.time_frequency_freq_min,
                      cfg.time_frequency_freq_max)

    time_frequency_cycles = cfg.time_frequency_cycles
    if time_frequency_cycles is None:
        time_frequency_cycles = freqs / 3.

    for condition in cfg.time_frequency_conditions:
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True,
            n_cycles=time_frequency_cycles
        )

        condition_str = sanitize_cond_name(condition)
        power_fname_out = bids_path.copy().update(
            suffix=f'power+{condition_str}+tfr', extension='.h5')
        itc_fname_out = bids_path.copy().update(
            suffix=f'itc+{condition_str}+tfr', extension='.h5')

        power.save(power_fname_out, overwrite=True)
        itc.save(itc_fname_out, overwrite=True)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        time_frequency_conditions=config.time_frequency_conditions,
        analyze_channels=config.analyze_channels,
        spatial_filter=config.spatial_filter,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference(),
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        time_frequency_cycles=config.time_frequency_cycles,
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked
    )
    return cfg


def main():
    """Run Time-frequency decomposition."""
    if not config.time_frequency_conditions:
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func = parallel_func(run_time_frequency)
    logs = parallel(
        run_func(cfg=get_config(), subject=subject, session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
