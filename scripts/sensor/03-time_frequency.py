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

import numpy as np

import mne
from mne.utils import BunchConst
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run, sanitize_cond_name

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
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

    msg = f'Input: {fname_in}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(fname_in)
    if cfg.analyze_channels:
        # We special-case the average reference here.
        # See 02-sliding_estimator.py for more info.
        if 'eeg' in cfg.ch_types and cfg.eeg_reference == 'average':
            epochs.set_eeg_reference('average')
        else:
            epochs.apply_proj()
        epochs.pick(cfg.analyze_channels)

    freqs = np.arange(cfg.time_frequency_freq_min,
                      cfg.time_frequency_freq_max)
    n_cycles = freqs / 3.

    for condition in cfg.time_frequency_conditions:
        this_epochs = epochs[condition]
        power, itc = mne.time_frequency.tfr_morlet(
            this_epochs, freqs=freqs, return_itc=True, n_cycles=n_cycles
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
) -> BunchConst:
    cfg = BunchConst(
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
        time_frequency_freq_max=config.time_frequency_freq_max
    )
    return cfg


def main():
    """Run Time-frequency decomposition."""
    parallel, run_func, _ = parallel_func(run_time_frequency,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(), subject=subject, session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
