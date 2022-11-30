"""Group average at the sensor level.

The M/EEG-channel data are averaged for group averages.
"""

import os
import os.path as op
from collections import defaultdict
from typing import Optional, TypedDict, List, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

import mne
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    get_decoding_contrasts, get_all_contrasts,
)
from ..._decoding import _handle_csp_args
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._run import failsafe_run, save_logs
from ..._report import run_report_average_sensor


def average_evokeds(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> List[mne.Evoked]:
    # Container for all conditions:
    all_evokeds = defaultdict(list)

    for this_subject in cfg.subjects:
        fname_in = BIDSPath(subject=this_subject,
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
        logger.info(**gen_log_kwargs(message=msg))

        evokeds = mne.read_evokeds(fname_in)
        for idx, evoked in enumerate(evokeds):
            all_evokeds[idx].append(evoked)  # Insert into the container

    for idx, evokeds in all_evokeds.items():
        all_evokeds[idx] = mne.grand_average(
            evokeds, interpolate_bads=cfg.interpolate_bads_grand_average
        )  # Combine subjects
        # Keep condition in comment
        all_evokeds[idx].comment = 'Grand average: ' + evokeds[0].comment

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
    logger.info(**gen_log_kwargs(message=msg))
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
    dtg_decim = cfg.decoding_time_generalization_decim
    if cfg.decoding_time_generalization and dtg_decim > 1:
        epochs.decimate(dtg_decim, verbose='error')
    times = epochs.times
    subjects = cfg.subjects
    del epochs, fname_epo

    for contrast in cfg.decoding_contrasts:
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
            'decim': dtg_decim,
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
                n_permutations=cfg.cluster_n_permutations,
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
    for contrast in cfg.decoding_contrasts:
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
        if not fname_out.fpath.parent.exists():
            os.makedirs(fname_out.fpath.parent)
        savemat(fname_out, contrast_score_stats)
        del contrast_score_stats, fname_out


def average_csp_decoding(
    cfg: SimpleNamespace,
    session: str,
    subject: str,
    condition_1: str,
    condition_2: str,
):
    msg = f'Summarizing CSP results: {condition_1} - {condition_2}.'
    logger.info(**gen_log_kwargs(message=msg))

    # Extract mean CV scores from all subjects.
    a_vs_b = f'{condition_1}+{condition_2}'.replace(op.sep, '')
    processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
    processing = processing.replace('_', '-').replace('-', '')

    all_decoding_data_freq = []
    all_decoding_data_time_freq = []

    # First load the data.
    fname_out = BIDSPath(
        subject='average',
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
    for subject in cfg.subjects:
        fname_xlsx = fname_out.copy().update(subject=subject)
        decoding_data_freq = pd.read_excel(
            fname_xlsx, sheet_name='CSP Frequency',
            dtype={'subject': str}  # don't drop trailing zeros
        )
        decoding_data_time_freq = pd.read_excel(
            fname_xlsx, sheet_name='CSP Time-Frequency',
            dtype={'subject': str}  # don't drop trailing zeros
        )
        all_decoding_data_freq.append(decoding_data_freq)
        all_decoding_data_time_freq.append(decoding_data_time_freq)
        del fname_xlsx

    # Now calculate descriptes and bootstrap CIs.
    grand_average_freq = _average_csp_time_freq(
        cfg=cfg,
        data=all_decoding_data_freq,
    )
    grand_average_time_freq = _average_csp_time_freq(
        cfg=cfg,
        data=all_decoding_data_time_freq,
    )

    with pd.ExcelWriter(fname_out) as w:
        grand_average_freq.to_excel(
            w, sheet_name='CSP Frequency', index=False
        )
        grand_average_time_freq.to_excel(
            w, sheet_name='CSP Time-Frequency', index=False
        )

    # Perform a cluster-based permutation test.
    subjects = cfg.subjects
    time_bins = np.array(cfg.decoding_csp_times)
    if time_bins.ndim == 1:
        time_bins = np.array(
            list(zip(time_bins[:-1], time_bins[1:]))
        )
    time_bins = pd.DataFrame(time_bins, columns=['t_min', 't_max'])
    freq_name_to_bins_map = _handle_csp_args(
        cfg.decoding_csp_times, cfg.decoding_csp_freqs, cfg.decoding_metric)
    data_for_clustering = {}
    for freq_range_name in freq_name_to_bins_map:
        a = np.empty(
            shape=(
                len(subjects),
                len(time_bins),
                len(freq_name_to_bins_map[freq_range_name])
            )
        )
        a.fill(np.nan)
        data_for_clustering[freq_range_name] = a

    g = (
        pd.concat(all_decoding_data_time_freq)
        .groupby([
            'subject', 'freq_range_name', 't_min', 't_max'
        ])
    )

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

    if cfg.cluster_forming_t_threshold is None:
        import scipy.stats
        cluster_forming_t_threshold = scipy.stats.t.ppf(
            1 - 0.05,  # one-sided test
            len(cfg.subjects) - 1
        )
    else:
        cluster_forming_t_threshold = cfg.cluster_forming_t_threshold

    cluster_permutation_results = {}
    for freq_range_name, X in data_for_clustering.items():
        t_vals, all_clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(  # noqa: E501
            X=X-0.5,  # One-sample test against zero.
            threshold=cluster_forming_t_threshold,
            n_permutations=cfg.cluster_n_permutations,
            adjacency=None,  # each time & freq bin connected to its neighbors
            out_type='mask',
            tail=1,  # one-sided: significantly above chance level
            seed=cfg.random_state,
        )
        n_permutations = H0.size - 1
        all_clusters = np.array(all_clusters)  # preserve "empty" 0th dimension
        cluster_permutation_results[freq_range_name] = {
            'mean_crossval_scores': X.mean(axis=0),
            't_vals': t_vals,
            'clusters': all_clusters,
            'cluster_p_vals': cluster_p_vals,
            'cluster_t_threshold': cluster_forming_t_threshold,
            'n_permutations': n_permutations,
            'time_bin_edges': cfg.decoding_csp_times,
            'freq_bin_edges': cfg.decoding_csp_freqs[freq_range_name],
        }

    fname_out.update(extension='.mat')
    savemat(file_name=fname_out, mdict=cluster_permutation_results)


def _average_csp_time_freq(
    *,
    cfg: SimpleNamespace,
    data: pd.DataFrame,
) -> pd.DataFrame:
    # Prepare a dataframe for storing the results.
    grand_average = data[0].copy()
    del grand_average['mean_crossval_score']

    grand_average['subject'] = 'average'
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

        # Abort here if we only have a single subject – no need to bootstrap
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
    *,
    config,
) -> SimpleNamespace:
    dtg_decim = config.decoding_time_generalization_decim
    cfg = SimpleNamespace(
        subjects=get_subjects(config),
        task=get_task(config),
        task_is_rest=config.task_is_rest,
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        proc=config.proc,
        deriv_root=config.deriv_root,
        conditions=config.conditions,
        contrasts=get_all_contrasts(config),
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_n_splits=config.decoding_n_splits,
        decoding_time_generalization=config.decoding_time_generalization,
        decoding_time_generalization_decim=dtg_decim,
        decoding_csp=config.decoding_csp,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        decoding_contrasts=get_decoding_contrasts(config),
        random_state=config.random_state,
        n_boot=config.n_boot,
        cluster_forming_t_threshold=config.cluster_forming_t_threshold,
        cluster_n_permutations=config.cluster_n_permutations,
        analyze_channels=config.analyze_channels,
        interpolate_bads_grand_average=config.interpolate_bads_grand_average,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        sessions=get_sessions(config),
        bids_root=config.bids_root,
        data_type=config.data_type,
        exclude_subjects=config.exclude_subjects,
        all_contrasts=get_all_contrasts(config),
        report_evoked_n_time_points=config.report_evoked_n_time_points,
        cluster_permutation_p_threshold=config.cluster_permutation_p_threshold,
    )
    return cfg


@failsafe_run()
def run_group_average_sensor(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
) -> None:
    if cfg.task_is_rest:
        msg = '    … skipping: for "rest" task.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    sessions = get_sessions(cfg)
    if not sessions:
        sessions = [None]

    with get_parallel_backend(exec_params):
        for session in sessions:
            evokeds = average_evokeds(
                cfg=cfg,
                subject=subject,
                session=session,
            )
            if exec_params.interactive:
                for evoked in evokeds:
                    evoked.plot()

            if cfg.decode:
                average_full_epochs_decoding(cfg, session)
                average_time_by_time_decoding(cfg, session)
        if cfg.decode and cfg.decoding_csp:
            parallel, run_func = parallel_func(
                average_csp_decoding, exec_params=exec_params)
            parallel(
                run_func(
                    cfg=cfg,
                    session=session,
                    subject=subject,
                    condition_1=contrast[0],
                    condition_2=contrast[1]
                )
                for session in get_sessions(config=cfg)
                for contrast in get_decoding_contrasts(config=cfg)
            )

        for session in sessions:
            run_report_average_sensor(
                cfg=cfg,
                exec_params=exec_params,
                subject=subject,
                session=session,
            )


def main(*, config: SimpleNamespace) -> None:
    log = run_group_average_sensor(
        cfg=get_config(
            config=config,
        ),
        exec_params=config.exec_params,
        subject='average',
    )
    save_logs(config=config, logs=[log])
