"""
Decoding based on common spatial patterns (CSP).
"""

import os.path as op
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from mne import BaseEpochs
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne_bids import BIDSPath
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    get_deriv_root, _restrict_analyze_channels, get_decoding_contrasts,
)
from ..._decoding import LogReg
from ..._logging import logger, gen_log_kwargs
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run, _script_path, save_logs


def _prepare_labels(
    *,
    epochs: BaseEpochs,
    contrast: Tuple[str, str]
) -> np.ndarray:
    """Return the projection of the events_id on a boolean vector.

    This projection is useful in the case of hierarchical events:
    we project the different events contained in one condition into
    just one label.

    Returns:
    --------
    A boolean numpy array containing the labels.
    """
    epochs_cond_0 = epochs[contrast[0]]
    event_codes_condition_0 = set(epochs_cond_0.events[:, 2])
    epochs_cond_1 = epochs[contrast[1]]
    event_codes_condition_1 = set(epochs_cond_1.events[:, 2])

    y = epochs.events[:, 2].copy()
    for i in range(len(y)):
        if y[i] in event_codes_condition_0 and y[i] in event_codes_condition_1:
            msg = (f"Event_id {y[i]} is contained both in "
                   f"{contrast[0]}'s set {event_codes_condition_0} and in "
                   f"{contrast[1]}'s set {event_codes_condition_1}."
                   f"{contrast} does not constitute a valid partition.")
            raise RuntimeError(msg)
        elif y[i] in event_codes_condition_0:
            y[i] = 0
        elif y[i] in event_codes_condition_1:
            y[i] = 1
        else:
            # This should not happen because epochs should already be filtered
            msg = (f"Event_id {y[i]} is not contained in "
                   f"{contrast[0]}'s set {event_codes_condition_0}  nor in "
                   f"{contrast[1]}'s set {event_codes_condition_1}.")
            raise RuntimeError(msg)
    return y


def prepare_epochs_and_y(
    *,
    epochs: BaseEpochs,
    contrast: Tuple[str, str],
    cfg,
    fmin: float,
    fmax: float
) -> Tuple[BaseEpochs, np.ndarray]:
    """Band-pass between, sub-select the desired epochs, and prepare y."""
    epochs_filt = (
        epochs
        .copy()
        .pick_types(
            meg=True, eeg=True,
        )
    )

    # We only take mag to speed up computation
    # because the information is redundant between grad and mag
    if cfg.datatype == 'meg' and cfg.use_maxwell_filter:
        epochs_filt.pick_types(meg='mag')

    # filtering out the conditions we are not interested in, to ensure here we
    # have a valid partition between the condition of the contrast.
    #
    # XXX Hack for handling epochs selection via metadata
    if contrast[0].startswith('event_name.isin'):
        epochs_filt = epochs_filt[f'{contrast[0]} or {contrast[1]}']
    else:
        epochs_filt = epochs_filt[contrast]

    # Filtering is costly, so do it last, after the selection of the channels
    # and epochs. We know that often the filter will be longer than the signal,
    # so we ignore the warning here.
    epochs_filt = epochs_filt.filter(fmin, fmax, n_jobs=1, verbose='error')
    y = _prepare_labels(epochs=epochs_filt, contrast=contrast)

    return epochs_filt, y


def get_input_fnames_csp(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    # TODO: Somehow remove this?
    del kwargs['contrast']
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    fname_epochs = BIDSPath(subject=subject,
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
    in_files = dict()
    in_files['epochs'] = fname_epochs
    return in_files


@failsafe_run(
    script_path=__file__,
    get_input_fnames=get_input_fnames_csp
)
def one_subject_decoding(
    *,
    cfg,
    subject: str,
    session: str,
    contrast: Tuple[str, str],
    in_files: Dict[str, BIDSPath]
) -> None:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.
    """
    condition1, condition2 = contrast
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(msg, subject=subject, session=session))

    bids_path = in_files['epochs'].copy().update(processing=None)
    epochs = mne.read_epochs(in_files.pop('epochs'))
    _restrict_analyze_channels(epochs, cfg)

    if cfg.time_frequency_subtract_evoked:
        epochs.subtract_evoked()

    # Perform rank reduction via PCA.
    #
    # Select the channel type with the smallest rank.
    # Limit it to a maximum of 100.
    ranks = mne.compute_rank(inst=epochs, rank='info')
    ch_type_smallest_rank = min(ranks, key=ranks.get)
    rank = min(
        ranks[ch_type_smallest_rank],
        100
    )
    del ch_type_smallest_rank, ranks

    msg = f'Reducing data dimension via PCA; new rank: {rank}.'
    logger.info(**gen_log_kwargs(msg, subject=subject, session=session))
    pca = UnsupervisedSpatialFilter(
        PCA(rank),
        average=False
    )

    # Classifier
    csp = CSP(
        n_components=4,  # XXX revisit
        reg=0.1,         # XXX revisit
        rank='info',
    )
    clf = make_pipeline(
        csp,
        LogReg(
            solver='liblinear',  # much faster than the default
            random_state=cfg.random_state,
            n_jobs=1,
        )
    )
    cv = StratifiedKFold(
        n_splits=cfg.decoding_n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    # Loop over frequencies (all time points lumped together)
    freq_name_to_bins_map = dict()
    for freq_range_name, freq_range_edges in cfg.decoding_csp_freqs.items():
        freq_bins = list(zip(freq_range_edges[:-1], freq_range_edges[1:]))
        freq_name_to_bins_map[freq_range_name] = freq_bins

    freq_decoding_table_rows = []
    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        for freq_bin in freq_bins:
            f_min, f_max = freq_bin
            row = {
                'subject': [subject],
                'cond_1': [condition1],
                'cond_2': [condition2],
                'f_min': [f_min],
                'f_max': [f_max],
                'freq_range_name': [freq_range_name],
                'mean_crossval_score': [np.nan],
                'scores': [np.ones(5)],
                'metric': [cfg.decoding_metric],
            }
            freq_decoding_table_rows.append(row)

    freq_decoding_table = pd.concat(
        [pd.DataFrame.from_dict(row) for row in freq_decoding_table_rows],
        ignore_index=True
    )
    del freq_decoding_table_rows

    def _fmt_contrast(cond1, cond2, fmin, fmax, freq_range_name,
                      tmin=None, tmax=None):
        msg = (
            f'Contrast: {cond1} – {cond2}, '
            f'{fmin:4.1f}–{fmax:4.1f} Hz ({freq_range_name})'
        )
        if tmin is not None:
            msg += f' {tmin:+5.3f}–{tmax:+5.3f} sec'
        return msg

    for idx, row in freq_decoding_table.iterrows():
        fmin = row['f_min']
        fmax = row['f_max']
        cond1 = row['cond_1']
        cond2 = row['cond_2']
        freq_range_name = row['freq_range_name']

        msg = _fmt_contrast(cond1, cond2, fmin, fmax, freq_range_name)
        logger.info(
            **gen_log_kwargs(msg, subject=subject, session=session)
        )

        # XXX We're filtering here again in each iteration. This should be
        # XXX optimized.
        epochs_filt, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg
        )
        # Get the data for all time points
        X = epochs_filt.get_data()

        # We apply PCA before running CSP:
        # - much faster CSP processing
        # - reduced risk of numerical instabilities.
        X_pca = pca.fit_transform(X)
        del X

        cv_scores = cross_val_score(
            estimator=clf,
            X=X_pca,
            y=y,
            scoring=cfg.decoding_metric,
            cv=cv,
            n_jobs=1,
        )
        freq_decoding_table.loc[idx, 'mean_crossval_score'] = cv_scores.mean()
        freq_decoding_table.at[idx, 'scores'] = cv_scores

    # Loop over times x frequencies
    #
    # Note: We don't support varying time ranges for different frequency
    # ranges to avoid leaking of information.
    time_bins = np.array(cfg.decoding_csp_times)
    if time_bins.ndim == 1:
        time_bins = np.array(
            list(zip(time_bins[:-1], time_bins[1:]))
        )
    assert time_bins.ndim == 2

    tf_decoding_table_rows = []

    for freq_range_name, freq_bins in freq_name_to_bins_map.items():
        for time_bin in time_bins:
            t_min, t_max = time_bin

            for freq_bin in freq_bins:
                f_min, f_max = freq_bin
                row = {
                    'subject': [subject],
                    'cond_1': [condition1],
                    'cond_2': [condition2],
                    't_min': [t_min],
                    't_max': [t_max],
                    'f_min': [f_min],
                    'f_max': [f_max],
                    'freq_range_name': [freq_range_name],
                    'mean_crossval_score': [np.nan],
                    'scores': [np.ones(5, dtype=float)],
                    'metric': [cfg.decoding_metric]
                }
                tf_decoding_table_rows.append(row)

    tf_decoding_table = pd.concat(
        [pd.DataFrame.from_dict(row) for row in tf_decoding_table_rows],
        ignore_index=True
    )
    del tf_decoding_table_rows

    for idx, row in tf_decoding_table.iterrows():
        tmin = row['t_min']
        tmax = row['t_max']
        fmin = row['f_min']
        fmax = row['f_max']
        cond1 = row['cond_1']
        cond2 = row['cond_2']
        freq_range_name = row['freq_range_name']

        epochs_filt, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg
        )
        # Crop data to the time window of interest
        if tmax is not None:  # avoid warnings about outside the interval
            tmax = min(tmax, epochs_filt.times[-1])
        epochs_filt.crop(tmin, tmax)
        X = epochs_filt.get_data()
        X_pca = pca.transform(X)
        del X

        cv_scores = cross_val_score(
            estimator=clf,
            X=X_pca,
            y=y,
            scoring=cfg.decoding_metric,
            cv=cv,
            n_jobs=1,
        )
        score = cv_scores.mean()
        tf_decoding_table.loc[idx, 'mean_crossval_score'] = score
        tf_decoding_table.at[idx, 'scores'] = cv_scores
        msg = _fmt_contrast(
            cond1, cond2, fmin, fmax, freq_range_name, tmin, tmax)
        msg += f': {cfg.decoding_metric}={score:0.3f}'
        logger.info(
            **gen_log_kwargs(msg, subject=subject, session=session)
        )

    # Write each DataFrame to a different Excel worksheet.
    a_vs_b = f'{condition1}+{condition2}'.replace(op.sep, '')
    processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
    processing = processing.replace('_', '-').replace('-', '')

    fname_results = bids_path.copy().update(suffix='decoding',
                                            processing=processing,
                                            extension='.xlsx')
    with pd.ExcelWriter(fname_results) as w:
        freq_decoding_table.to_excel(
            w, sheet_name='CSP Frequency', index=False
        )
        tf_decoding_table.to_excel(
            w, sheet_name='CSP Time-Frequency', index=False
        )

    assert len(in_files) == 0, in_files.keys()

    out_files = {'csp-excel': fname_results}
    return out_files


def get_config(
    *,
    config,
    subject: str,
    session: Optional[str]
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        # Data parameters
        datatype=get_datatype(config),
        deriv_root=get_deriv_root(config),
        task=get_task(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        use_maxwell_filter=config.use_maxwell_filter,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        # Processing parameters
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked,
        decoding_metric=config.decoding_metric,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        decoding_n_splits=config.decoding_n_splits,
        n_boot=config.n_boot,
        random_state=config.random_state,
        interactive=config.interactive
    )
    return cfg


def main():
    """Run all subjects decoding in parallel."""
    import config
    if not config.contrasts or not config.decoding_csp:
        if not config.contrasts:
            msg = 'No contrasts specified. '
        else:
            msg = 'No CSP analysis requested. '

        msg = 'Skipping …'
        with _script_path(__file__):
            logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config):
        parallel, run_func = parallel_func(one_subject_decoding, config=config)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session
                ),
                subject=subject,
                session=session,
                contrast=contrast,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for contrast in get_decoding_contrasts(config)
        )
        save_logs(logs=logs, config=config)


if __name__ == '__main__':
    main()
