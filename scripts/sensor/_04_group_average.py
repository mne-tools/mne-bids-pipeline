"""
=====================================
07. Group average at the sensor level
=====================================

The M/EEG-channel data are averaged for group averages.
"""

import os
import os.path as op
from collections import defaultdict
import logging
from typing import Optional
from types import SimpleNamespace

from tqdm import tqdm
import numpy as np
from scipy.io import loadmat, savemat

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


def average_evokeds(cfg, session):
    # Container for all conditions:
    all_evokeds = defaultdict(list)

    for subject in cfg.subjects:
        fname_in = BIDSPath(subject=subject,
                            session=session,
                            task=cfg.task,
                            acquisition=cfg.acq,
                            run=None,
                            recording=cfg.rec,
                            space=cfg.space,
                            suffix='ave',
                            extension='.fif',
                            datatype=cfg.datatype,
                            root=cfg.deriv_root,
                            check=False)

        msg = f'Input: {fname_in.basename}'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

        evokeds = mne.read_evokeds(fname_in)
        for idx, evoked in enumerate(evokeds):
            all_evokeds[idx].append(evoked)  # Insert into the container

    for idx, evokeds in all_evokeds.items():
        all_evokeds[idx] = mne.grand_average(
            evokeds, interpolate_bads=cfg.interpolate_bads_grand_average
        )  # Combine subjects
        # Keep condition in comment
        all_evokeds[idx].comment = 'Grand average: ' + evokeds[0].comment

    subject = 'average'
    fname_out = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         processing=cfg.proc,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix='ave',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    if not fname_out.fpath.parent.exists():
        os.makedirs(fname_out.fpath.parent)

    msg = f'Saving grand-averaged evoked sensor data: {fname_out.basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    mne.write_evokeds(fname_out, list(all_evokeds.values()), overwrite=True)
    return list(all_evokeds.values())


def average_decoding(cfg, session):
    # Get the time points from the very first subject. They are identical
    # across all subjects and conditions, so this should suffice.
    fname_epo = BIDSPath(subject=cfg.subjects[0],
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         suffix='epo',
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)
    epochs = mne.read_epochs(fname_epo)
    times = epochs.times
    subjects = cfg.subjects
    del epochs, fname_epo

    for contrast in cfg.contrasts:
        cond_1, cond_2 = contrast
        contrast_score_stats = {'cond_1': cond_1,
                                'cond_2': cond_2,
                                'times': times,
                                'N': len(subjects),
                                'mean': np.empty(len(times)),
                                'mean_min': np.empty(len(times)),
                                'mean_max': np.empty(len(times)),
                                'mean_se': np.empty(len(times)),
                                'mean_ci_lower': np.empty(len(times)),
                                'mean_ci_upper': np.empty(len(times))}

        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        processing = f'{a_vs_b}+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')

        # Extract mean CV scores from all subjects.
        mean_scores = np.empty((len(subjects), len(times)))
        for sub_idx, subject in enumerate(subjects):
            fname_mat = BIDSPath(subject=subject,
                                 session=session,
                                 task=cfg.task,
                                 acquisition=cfg.acq,
                                 run=None,
                                 recording=cfg.rec,
                                 space=cfg.space,
                                 processing=processing,
                                 suffix='decoding',
                                 extension='.mat',
                                 datatype=cfg.datatype,
                                 root=cfg.deriv_root,
                                 check=False)

            decoding_data = loadmat(fname_mat)
            mean_scores[sub_idx, :] = decoding_data['scores'].mean(axis=0)

        # Now we can calculate some descriptive statistics on the mean scores.
        # We use the [:] here as a safeguard to ensure we don't mess up the
        # dimensions.
        contrast_score_stats['mean'][:] = mean_scores.mean(axis=0)
        contrast_score_stats['mean_min'][:] = mean_scores.min(axis=0)
        contrast_score_stats['mean_max'][:] = mean_scores.max(axis=0)

        # Finally, for each time point, bootstrap the mean, and calculate the
        # SD of the bootstrapped distribution: this is the standard error of
        # the mean. We also derive 95% confidence intervals.
        rng = np.random.default_rng(seed=cfg.random_state)

        for time_idx in tqdm(range(len(times)), desc='Bootstrapping means'):
            scores_resampled = rng.choice(mean_scores[:, time_idx],
                                          size=(cfg.n_boot, len(subjects)),
                                          replace=True)
            bootstrapped_means = scores_resampled.mean(axis=1)

            # SD of the bootstrapped distribution == SE of the metric.
            se = bootstrapped_means.std(ddof=1)
            ci_lower = np.quantile(bootstrapped_means, q=0.025)
            ci_upper = np.quantile(bootstrapped_means, q=0.975)

            contrast_score_stats['mean_se'][time_idx] = se
            contrast_score_stats['mean_ci_lower'][time_idx] = ci_lower
            contrast_score_stats['mean_ci_upper'][time_idx] = ci_upper

            del bootstrapped_means, se, ci_lower, ci_upper

        fname_out = fname_mat.copy().update(subject='average')
        savemat(fname_out, contrast_score_stats)
        del contrast_score_stats, fname_out


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        subjects=config.get_subjects(),
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        deriv_root=config.get_deriv_root(),
        conditions=config.conditions,
        contrasts=config.get_decoding_contrasts(),
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_n_splits=config.decoding_n_splits,
        random_state=config.random_state,
        n_boot=config.n_boot,
        analyze_channels=config.analyze_channels,
        interpolate_bads_grand_average=config.interpolate_bads_grand_average,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference(),
        interactive=config.interactive
    )
    return cfg


# pass 'average' subject for logging
@failsafe_run(on_error=on_error, script_path=__file__)
def run_group_average_sensor(*, cfg, subject='average'):
    if config.get_task().lower() == 'rest':
        msg = '    … skipping: for "rest" task.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    sessions = config.get_sessions()
    if not sessions:
        sessions = [None]

    for session in sessions:
        evokeds = average_evokeds(cfg, session)
        if config.interactive:
            for evoked in evokeds:
                evoked.plot()

        if config.decode:
            average_decoding(cfg, session)


def main():
    log = run_group_average_sensor(cfg=get_config(), subject='average')
    config.save_logs([log])


if __name__ == '__main__':
    main()
