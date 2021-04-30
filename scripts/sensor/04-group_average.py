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
from tqdm import tqdm

import numpy as np
from scipy.io import loadmat, savemat

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_message

logger = logging.getLogger('mne-bids-pipeline')


msg = 'Running Step 9: Grand-average sensor data'
logger.info(gen_log_message(step=9, message=msg))


def average_evokeds(session):
    # Container for all conditions:
    all_evokeds = defaultdict(list)

    for subject in config.get_subjects():
        fname_in = BIDSPath(subject=subject,
                            session=session,
                            task=config.get_task(),
                            acquisition=config.acq,
                            run=None,
                            recording=config.rec,
                            space=config.space,
                            suffix='ave',
                            extension='.fif',
                            datatype=config.get_datatype(),
                            root=config.get_deriv_root(),
                            check=False)

        msg = f'Input: {fname_in}'
        logger.info(gen_log_message(message=msg, step=9, subject=subject,
                                    session=session))

        evokeds = mne.read_evokeds(fname_in)
        for idx, evoked in enumerate(evokeds):
            all_evokeds[idx].append(evoked)  # Insert into the container

    for idx, evokeds in all_evokeds.items():
        all_evokeds[idx] = mne.grand_average(
            evokeds, interpolate_bads=config.interpolate_bads_grand_average
        )  # Combine subjects

    subject = 'average'
    fname_out = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         processing=config.proc,
                         recording=config.rec,
                         space=config.space,
                         suffix='ave',
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root(),
                         check=False)

    if not fname_out.fpath.parent.exists():
        os.makedirs(fname_out.fpath.parent)

    msg = f'Saving grand-averaged evoked sensor data: {fname_out}'
    logger.info(gen_log_message(message=msg, step=9, subject=subject,
                                session=session))
    mne.write_evokeds(fname_out, list(all_evokeds.values()))
    return list(all_evokeds.values())


def average_decoding(session):
    # Get the time points from the very first subject. They are identical
    # across all subjects and conditions, so this should suffice.
    fname_epo = BIDSPath(subject=config.get_subjects()[0],
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         suffix='epo',
                         extension='.fif',
                         datatype=config.get_datatype(),
                         root=config.get_deriv_root(),
                         check=False)
    epochs = mne.read_epochs(fname_epo)
    times = epochs.times
    subjects = config.get_subjects()
    del epochs, fname_epo

    for contrast in config.contrasts:
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
        processing = f'{a_vs_b}+{config.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')

        # Extract mean CV scores from all subjects.
        mean_scores = np.empty((len(subjects), len(times)))
        for sub_idx, subject in enumerate(subjects):
            fname_mat = BIDSPath(subject=subject,
                                 session=session,
                                 task=config.get_task(),
                                 acquisition=config.acq,
                                 run=None,
                                 recording=config.rec,
                                 space=config.space,
                                 processing=processing,
                                 suffix='decoding',
                                 extension='.mat',
                                 datatype=config.get_datatype(),
                                 root=config.get_deriv_root(),
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
        rng = np.random.default_rng(seed=config.random_state)

        for time_idx in tqdm(range(len(times)), desc='Bootstrapping means'):
            scores_resampled = rng.choice(mean_scores[:, time_idx],
                                          size=(config.n_boot, len(subjects)),
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


def main():
    sessions = config.get_sessions()
    if not sessions:
        sessions = [None]

    for session in sessions:
        evokeds = average_evokeds(session)
        if config.interactive:
            for evoked in evokeds:
                evoked.plot()

        if config.decode:
            average_decoding(session)


if __name__ == '__main__':
    main()


msg = 'Completed Step 9: Grand-average sensor data'
logger.info(gen_log_message(step=9, message=msg))
