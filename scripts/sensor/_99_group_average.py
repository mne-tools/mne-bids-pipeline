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
from typing import Optional, TypedDict, List, Tuple, Literal
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


class ClusterAcrossTime(TypedDict):
    times: np.ndarray
    p_value: float


def _decoding_cluster_permutation_test(
    scores: np.ndarray,
    times: np.ndarray,
    cluster_forming_t_threshold: Optional[float],
    n_permutations: int,
    random_seed: int
) -> Tuple[
    np.ndarray, List[ClusterAcrossTime], int
]:
    """Perform a cluster permutation test on decoding scores.

    The clusters are formed across time points.
    """
    t_vals, all_clusters, cluster_p_vals, H0 = \
        mne.stats.permutation_cluster_1samp_test(
            X=scores,
            threshold=cluster_forming_t_threshold,
            n_permutations=n_permutations,
            adjacency=None,  # each time point is "connected" to its neighbors
            out_type='mask',
            tail=1,  # one-sided: significantly above chance level
            seed=random_seed,
            verbose=True
        )
    n_permutations = H0.size - 1

    # Convert to a list of Clusters
    clusters = []
    for cluster_idx, cluster_time_slice in enumerate(all_clusters):
        cluster_times = times[cluster_time_slice]
        cluster_p_val = cluster_p_vals[cluster_idx]
        cluster = ClusterAcrossTime(
            times=cluster_times,
            p_value=cluster_p_val
        )
        clusters.append(cluster)

    return t_vals, clusters, n_permutations


def average_time_by_time_decoding(
    cfg: SimpleNamespace,
    session: str
):
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
        if cfg.decoding_time_generalization:
            time_points_shape = (len(times), len(times))
        else:
            time_points_shape = (len(times),)

        contrast_score_stats = {
            'cond_1': cond_1,
            'cond_2': cond_2,
            'times': times,
            'N': len(subjects),
            'mean': np.empty(time_points_shape),
            'mean_min': np.empty(time_points_shape),
            'mean_max': np.empty(time_points_shape),
            'mean_se': np.empty(time_points_shape),
            'mean_ci_lower': np.empty(time_points_shape),
            'mean_ci_upper': np.empty(time_points_shape),
            'cluster_all_times': np.array([]),
            'cluster_all_t_values': np.array([]),
            'cluster_t_threshold': np.nan,
            'cluster_n_permutations': np.nan,
            'clusters': list()
        }

        processing = (f'{cond_1}+{cond_2}+TimeByTime+{cfg.decoding_metric}'
                      .replace(op.sep, '')
                      .replace('_', '-')
                      .replace('-', ''))

        # Extract mean CV scores from all subjects.
        mean_scores = np.empty((len(subjects), *time_points_shape))

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

        # Cluster permutation test.
        # We can only permute for two or more subjects
        #
        # If we've performed time generalization, we will only use the diagonal
        # CV scores here (classifiers trained and tested at the same time
        # points).

        if len(subjects) > 1:
            # Constrain cluster permutation test to time points of the
            # time-locked event or later.
            # We subtract the chance level from the scores as we'll be
            # performing a 1-sample test (i.e., test against 0)!
            idx = np.where(times >= 0)[0]

            if cfg.decoding_time_generalization:
                cluster_permutation_scores = mean_scores[:, idx, idx] - 0.5
            else:
                cluster_permutation_scores = mean_scores[:, idx] - 0.5

            cluster_permutation_times = times[idx]
            if cfg.cluster_forming_t_threshold is None:
                import scipy.stats
                cluster_forming_t_threshold = scipy.stats.t.ppf(
                    1 - 0.05,
                    len(cluster_permutation_scores) - 1
                )
            else:
                cluster_forming_t_threshold = cfg.cluster_forming_t_threshold

            t_vals, clusters, n_perm = _decoding_cluster_permutation_test(
                scores=cluster_permutation_scores,
                times=cluster_permutation_times,
                cluster_forming_t_threshold=cluster_forming_t_threshold,
                n_permutations=cfg.n_permutations,
                random_seed=cfg.random_state
            )

            contrast_score_stats.update({
                'cluster_all_times': cluster_permutation_times,
                'cluster_all_t_values': t_vals,
                'cluster_t_threshold': cluster_forming_t_threshold,
                'clusters': clusters,
                'cluster_n_permutations': n_perm
            })

            del cluster_permutation_scores, cluster_permutation_times, n_perm

        # Now we can calculate some descriptive statistics on the mean scores.
        # We use the [:] here as a safeguard to ensure we don't mess up the
        # dimensions.
        #
        # For time generalization, all values (each time point vs each other)
        # are considered.
        contrast_score_stats['mean'][:] = mean_scores.mean(axis=0)
        contrast_score_stats['mean_min'][:] = mean_scores.min(axis=0)
        contrast_score_stats['mean_max'][:] = mean_scores.max(axis=0)

        # Finally, for each time point, bootstrap the mean, and calculate the
        # SD of the bootstrapped distribution: this is the standard error of
        # the mean. We also derive 95% confidence intervals.
        rng = np.random.default_rng(seed=cfg.random_state)
        for time_idx in range(len(times)):
            if cfg.decoding_time_generalization:
                data = mean_scores[:, time_idx, time_idx]
            else:
                data = mean_scores[:, time_idx]
            scores_resampled = rng.choice(data,
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


def average_full_epochs_decoding(
    cfg: SimpleNamespace,
    session: str
):
    for contrast in cfg.contrasts:
        cond_1, cond_2 = contrast
        n_subjects = len(cfg.subjects)

        contrast_score_stats = {
            'cond_1': cond_1,
            'cond_2': cond_2,
            'N': n_subjects,
            'subjects': cfg.subjects,
            'scores': np.nan,
            'mean': np.nan,
            'mean_min': np.nan,
            'mean_max': np.nan,
            'mean_se': np.nan,
            'mean_ci_lower': np.nan,
            'mean_ci_upper': np.nan,
        }

        processing = (f'{cond_1}+{cond_2}+FullEpochs+{cfg.decoding_metric}'
                      .replace(op.sep, '')
                      .replace('_', '-')
                      .replace('-', ''))

        # Extract mean CV scores from all subjects.
        mean_scores = np.empty(n_subjects)
        for sub_idx, subject in enumerate(cfg.subjects):
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
            mean_scores[sub_idx] = decoding_data['scores'].mean()

        # Now we can calculate some descriptive statistics on the mean scores.
        # We use the [:] here as a safeguard to ensure we don't mess up the
        # dimensions.
        contrast_score_stats['scores'] = mean_scores
        contrast_score_stats['mean'] = mean_scores.mean()
        contrast_score_stats['mean_min'] = mean_scores.min()
        contrast_score_stats['mean_max'] = mean_scores.max()

        # Finally, bootstrap the mean, and calculate the
        # SD of the bootstrapped distribution: this is the standard error of
        # the mean. We also derive 95% confidence intervals.
        rng = np.random.default_rng(seed=cfg.random_state)
        scores_resampled = rng.choice(mean_scores,
                                      size=(cfg.n_boot, n_subjects),
                                      replace=True)
        bootstrapped_means = scores_resampled.mean(axis=1)

        # SD of the bootstrapped distribution == SE of the metric.
        se = bootstrapped_means.std(ddof=1)
        ci_lower = np.quantile(bootstrapped_means, q=0.025)
        ci_upper = np.quantile(bootstrapped_means, q=0.975)

        contrast_score_stats['mean_se'] = se
        contrast_score_stats['mean_ci_lower'] = ci_lower
        contrast_score_stats['mean_ci_upper'] = ci_upper

        del bootstrapped_means, se, ci_lower, ci_upper

        fname_out = fname_mat.copy().update(subject='average')
        savemat(fname_out, contrast_score_stats)
        del contrast_score_stats, fname_out


def average_csp_decoding(
    cfg: SimpleNamespace,
    session: str
):
    for contrast in cfg.contrasts:
        cond_1, cond_2 = contrast


        # Extract mean CV scores from all subjects.
        a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
        processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')

        all_decoding_data_freq = []
        all_decoding_data_time_freq = []

        # First load the data.
        for subject in cfg.subjects:
            fname_xlsx = BIDSPath(
                subject=subject,
                session=session,
                task=cfg.task,
                acquisition=cfg.acq,
                run=None,
                recording=cfg.rec,
                space=cfg.space,
                processing=processing,
                suffix='decoding',
                extension='.xlsx',
                datatype=cfg.datatype,
                root=cfg.deriv_root,
                check=False
            )

            decoding_data_freq = pd.read_excel(
                fname_xlsx, sheet_name='CSP Frequency',
            )
            decoding_data_time_freq = pd.read_excel(
                fname_xlsx, sheet_name='CSP Time-Frequency'
            )
            all_decoding_data_freq.append(decoding_data_freq)
            all_decoding_data_time_freq.append(decoding_data_time_freq)


        # Now calculate descriptes and bootstrap CIs.
        grand_average_freq = _average_csp_time_freq(
            cfg=cfg,
            data=all_decoding_data_freq,
        )
        grand_average_time_freq = _average_csp_time_freq(
            cfg=cfg,
            data=all_decoding_data_time_freq,
        )

        fname_out = fname_xlsx.copy().update(subject='average')
        with pd.ExcelWriter(fname_out) as w:
            grand_average_freq.to_excel(
                w, sheet_name='CSP Frequency', index=False
            )
            grand_average_time_freq.to_excel(
                w, sheet_name='CSP Time-Frequency', index=False
            )

        # Perform a cluster-based permutation test.
        g = (
            pd.concat(all_decoding_data_time_freq)
            .groupby([
                'subject', 'freq_range_name', 't_min', 't_max'
            ])
        )

        subjects = list(
            pd.concat(all_decoding_data_time_freq)['subject'].unique()
        )
        time_bins = np.array(cfg.decoding_csp_times)
        if time_bins.ndim == 1:
            time_bins = np.array(
                list(zip(time_bins[:-1], time_bins[1:]))
            )
        time_bins = pd.DataFrame(time_bins, columns=['t_min', 't_max'])
        freq_range_names = list(
            pd.concat(all_decoding_data_time_freq)['freq_range_name'].unique()
        )
        freq_name_to_bins_map = dict()
        for freq_range_name, freq_range_edges in cfg.decoding_csp_freqs.items():
            freq_bins = list(zip(freq_range_edges[:-1], freq_range_edges[1:]))
            freq_name_to_bins_map[freq_range_name] = freq_bins

        data_for_clustering = {}
        for freq_range_name in freq_range_names:
            a = np.empty(
                shape=(
                    len(subjects),
                    len(time_bins),
                    len(freq_name_to_bins_map[freq_range_name])
                )
            )
            a.fill(np.nan)
            data_for_clustering[freq_range_name] = a

        for (subject, freq_range_name, t_min, t_max), df in g:
            scores = df['mean_crossval_score']
            sub_idx = subjects.index(subject)
            time_bin_idx = time_bins.loc[
                (np.isclose(time_bins['t_min'], t_min)) &
                (np.isclose(time_bins['t_max'], t_max)), :
            ].index
            assert len(time_bin_idx) == 1
            time_bin_idx = time_bin_idx[0]
            data_for_clustering[freq_range_name][sub_idx][time_bin_idx] = scores

        cluster_permutation_results = {}
        for freq_range_name, X in data_for_clustering.items():
            X = X - 0.5  # One-sample test against zero.
            t_vals, all_clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(  # noqa: E501
                X=X,
                threshold=cfg.cluster_forming_t_threshold,
                n_permutations=cfg.n_permutations,
                tail=0,
                seed=cfg.random_state,
                adjacency=None,
                out_type='mask',
            )
            n_permutations = H0.size - 1
            cluster_permutation_results[freq_range_name] = {
                't_vals': t_vals,
                'clusters': all_clusters,
                'cluster_p_vals': cluster_p_vals,
                'n_permutations': n_permutations,
                'time_bin_edges': cfg.decoding_csp_times,
                'freq_bin_edges': cfg.decoding_csp_freqs[freq_range_name],
            }

        fname_out = fname_xlsx.copy().update(subject='average',
                                             extension='.mat')
        savemat(file_name=fname_out, mdict=cluster_permutation_results)


def _average_csp_time_freq(
    *,
    cfg: SimpleNamespace,
    data: pd.DataFrame,
) -> pd.DataFrame:
    # Prepare a dataframe for storing the results.
    grand_average = data[0].copy()
    del grand_average['mean_crossval_score']

    grand_average['mean'] = np.nan
    grand_average['mean_se'] = np.nan
    grand_average['mean_ci_lower'] = np.nan
    grand_average['mean_ci_upper'] = np.nan

    # Now generate descriptive and bootstrapped statistics.
    n_subjects = len(cfg.subjects)
    rng = np.random.default_rng(seed=cfg.random_state)
    for row_idx, row in grand_average.iterrows():
        all_scores = np.array([
            df.loc[row_idx, 'mean_crossval_score']
            for df in data
        ])

        grand_average.loc[row_idx, 'mean'] = all_scores.mean()

        # Abort here if we only have a single subject – no need to bootstrap
        # CIs etc.
        if len(cfg.subjects) == 1:
            continue

        # Bootstrap the mean, and calculate the
        # SD of the bootstrapped distribution: this is the standard error of
        # the mean. We also derive 95% confidence intervals.
        scores_resampled = rng.choice(
            all_scores,
            size=(cfg.n_boot, n_subjects),
            replace=True
        )
        bootstrapped_means = scores_resampled.mean(axis=1)

        # SD of the bootstrapped distribution == SE of the metric.
        se = bootstrapped_means.std(ddof=1)
        ci_lower = np.quantile(bootstrapped_means, q=0.025)
        ci_upper = np.quantile(bootstrapped_means, q=0.975)

        grand_average.loc[row_idx, 'mean_se'] = se
        grand_average.loc[row_idx, 'mean_ci_lower'] = ci_lower
        grand_average.loc[row_idx, 'mean_ci_upper'] = ci_upper

        del (
                bootstrapped_means, se, ci_lower, ci_upper, scores_resampled,
                all_scores, row_idx, row
            )

    return grand_average


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
        decoding_time_generalization=config.decoding_time_generalization,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        random_state=config.random_state,
        n_boot=config.n_boot,
        cluster_forming_t_threshold=config.cluster_forming_t_threshold,
        n_permutations=config.cluster_n_permutations,
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
            average_full_epochs_decoding(cfg, session)
            average_time_by_time_decoding(cfg, session)

            if config.decoding_csp:
                average_csp_decoding(cfg, session)


def main():
    log = run_group_average_sensor(cfg=get_config(), subject='average')
    config.save_logs([log])


if __name__ == '__main__':
    main()
