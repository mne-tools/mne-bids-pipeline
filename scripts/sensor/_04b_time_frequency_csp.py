
import itertools
from types import SimpleNamespace
from typing import Optional, Tuple
import logging
import os.path as op

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

import mne
from mne import BaseEpochs
from mne.decoding import UnsupervisedSpatialFilter, CSP
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


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
            logger.critical(msg)
        elif y[i] in event_codes_condition_0:
            y[i] = 0
        elif y[i] in event_codes_condition_1:
            y[i] = 1
        else:
            # This should not happen because epochs should already by filtered
            msg = (f"Event_id {y[i]} is not contained in "
                   f"{contrast[0]}'s set {event_codes_condition_0}  nor in "
                   f"{contrast[1]}'s set {event_codes_condition_1}.")
            logger.critical(msg)
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
            meg=True, eeg=True, stim=False, eog=False, exclude='bads'
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
    # and epochs.
    epochs_filt = epochs_filt.filter(fmin, fmax, n_jobs=1)
    y = _prepare_labels(epochs=epochs_filt, contrast=contrast)

    return epochs_filt, y


@failsafe_run(on_error=on_error, script_path=__file__)
def one_subject_decoding(
    *,
    cfg,
    subject: str,
    session: str,
    contrast: Tuple[str, str],

) -> None:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.
    """
    condition1, condition2 = contrast
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(msg, subject=subject, session=session))

    # XXX The following 30 or so lines are actually identical to code we have
    # in _03_decoding_time_by_time. We should try to refactor this to reduce
    # duplication.
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

    epochs = mne.read_epochs(fname_epochs)
    if cfg.analyze_channels:
        # We special-case the average reference here to work around a situation
        # where e.g. `analyze_channels` might contain only a single channel:
        # `concatenate_epochs` below will then fail when trying to create /
        # apply the projection. We can avoid this by removing an existing
        # average reference projection here, and applying the average reference
        # directly – without going through a projector.
        if 'eeg' in cfg.ch_types and cfg.eeg_reference == 'average':
            epochs.set_eeg_reference('average')
        else:
            epochs.apply_proj()
        epochs.pick(cfg.analyze_channels)

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
    clf = make_pipeline(csp, LinearDiscriminantAnalysis())
    cv = StratifiedKFold(
        n_splits=cfg.decoding_n_splits,
        shuffle=True,
        random_state=cfg.random_state
    )

    if isinstance(cfg.decoding_csp_freqs, dict):
        freq_range_names = list(cfg.decoding_csp_freqs.keys())
        freq_ranges = np.array(
            list(cfg.decoding_csp_freqs.values())
        )
    else:
        freq_ranges = np.array(cfg.decoding_csp_freqs)
        freq_range_names = None

    if freq_ranges.ndim == 1:
        freq_ranges = np.array(
            list(zip(freq_ranges[:-1], freq_ranges[1:]))
        )

    if freq_range_names is None:
        freq_range_names = [None] * len(freq_ranges)

    assert freq_ranges.ndim == 2
    assert len(freq_ranges) == len(freq_range_names)

    time_ranges = np.array(cfg.decoding_csp_times)
    if time_ranges.ndim == 1:
        time_ranges = np.array(
            list(zip(time_ranges[:-1], time_ranges[1:]))
        )
    assert time_ranges.ndim == 2

    # Roll covariance, csp and lda over time
    tf_scores = np.empty(
        (len(time_ranges), len(freq_ranges))
    )
    tf_scores.fill(np.nan)
    freq_scores = []

    for freq_range_idx, (freq_range, freq_range_name) in enumerate(
        zip(freq_ranges, freq_range_names)
    ):
        fmin, fmax = freq_range
        msg = (
            f'Contrast: {contrast[0]} – {contrast[1]}, '
            f'Freqs (Hz): {fmin}–{fmax}'
        )
        if freq_range_name is not None:
            msg += f' ({freq_range_name})'
        logger.info(
            **gen_log_kwargs(msg, subject=subject, session=session)
        )

        epochs_filt, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg
        )

        ######################################################################
        # 1. Loop over frequencies, apply classifier and save scores

        X = epochs_filt.get_data()

        # We apply PCA before running CSP:
        # - much faster CSP processing
        # - reduced risk of numerical instabilities.

        X_pca = pca.fit_transform(X)
        del X

        cv_scores = cross_val_score(
            estimator=clf, X=X_pca, y=y,
            scoring=cfg.decoding_metric,
            cv=cv,
            n_jobs=1
        )

        freq_scores.append(cv_scores.mean())

        ######################################################################
        # 2. Loop through frequencies and time
        for time_range_idx, time_range in enumerate(time_ranges):
            # Originally the window size varied accross frequencies...
            # But this means also that there is some mutual information between
            # 2 different pixels in the map.
            # So the simple way to deal with this is just to fix
            # the windows size for all frequencies.

            tmin, tmax = time_range
            msg = (
                f'Contrast: {contrast[0]} – {contrast[1]}, '
                f'Freqs (Hz): {fmin}–{fmax}'
            )
            if freq_range_name is not None:
                msg += f' ({freq_range_name})'
            msg += f', Times (s): {round(tmin, 3)}–{round(tmax, 3)}'
            logger.info(
                **gen_log_kwargs(msg, subject=subject, session=session)
            )

            # Crop data into time window of interest
            X = epochs_filt.copy().crop(tmin, tmax).get_data()
            X_pca = pca.transform(X)
            del X

            cv_scores = cross_val_score(
                estimator=clf,
                X=X_pca, y=y,
                scoring=cfg.decoding_metric,
                cv=cv,
                n_jobs=1,
            )
            tf_scores[time_range_idx, freq_range_idx] = cv_scores.mean()


    # Prepare frequency data for saving
    tabular_data_freq = pd.DataFrame(
        columns=['cond_1', 'cond_2', 'freq_range', 'freq_range_name'
                 'mean_crossval_score', 'metric']
    )
    for freq_range_idx, freq_range in enumerate(freq_ranges):
        freq_range_name = freq_range_names[freq_range_idx]
        one_row = pd.DataFrame(
            {
                'cond_1': condition1,
                'cond_2': condition2,
                'freq_range': f'{freq_range[0]}–{freq_range[1]}',
                'freq_range_name': freq_range_name if freq_range_name is not None else '',
                'mean_crossval_score': freq_scores[freq_range_idx],
                'metric': cfg.decoding_metric
            },
            index=[0]
        )
        tabular_data_freq = pd.concat(
            [tabular_data_freq, one_row],
            ignore_index=True
        )

    # Prepare time-frequency data for saving
    tabular_data_tf = pd.DataFrame(
        columns=['cond_1', 'cond_2', 'time_range', 'freq_range',
                 'freq_range_name', 'mean_crossval_score', 'metric']
    )
    for time_range_idx, time_range in enumerate(time_ranges):
        for freq_range_idx, freq_range in enumerate(freq_ranges):
            freq_range_name = freq_range_names[freq_range_idx]
            one_row = pd.DataFrame(
                {
                    'cond_1': condition1,
                    'cond_2': condition2,
                    'time_range': f'{time_range[0]}–{time_range[1]}',
                    'freq_range': f'{freq_range[0]}–{freq_range[1]}',
                    'freq_range_name': freq_range_name if freq_range_name is not None else '',
                    'mean_crossval_score': tf_scores[time_range_idx, freq_range_idx],
                    'metric': cfg.decoding_metric
                },
                index=[0]
            )
            tabular_data_tf = pd.concat(
                [tabular_data_tf, one_row],
                ignore_index=True
            )

    # Write each DataFrame to a different Excel worksheet.
    a_vs_b = f'{condition1}+{condition2}'.replace(op.sep, '')
    processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
    processing = processing.replace('_', '-').replace('-', '')

    fname_results = fname_epochs.copy().update(suffix='decoding',
                                               processing=processing,
                                               extension='.xlsx')

    with pd.ExcelWriter(fname_results) as w:
        tabular_data_freq.to_excel(w, sheet_name='CSP Frequency', index=False)
        tabular_data_tf.to_excel(
            w, sheet_name='CSP Time-Frequency', index=False
        )


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        # Data parameters
        datatype=config.get_datatype(),
        deriv_root=config.get_deriv_root(),
        task=config.get_task(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        use_maxwell_filter=config.use_maxwell_filter,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference(),
        # Processing parameters
        time_frequency_subtract_evoked=config.time_frequency_subtract_evoked,
        decoding_metric=config.decoding_metric,
        decoding_csp_freqs=config.decoding_csp_freqs,
        decoding_csp_times=config.decoding_csp_times,
        decoding_n_splits=config.decoding_n_splits,
        csp_plot_patterns=config.csp_plot_patterns,
        n_boot=config.n_boot,
        random_state=config.random_state,
        interactive=config.interactive
    )
    return cfg


def main():
    """Run all subjects decoding in parallel."""
    msg = 'Running Step 4b: CSP'
    logger.info(**gen_log_kwargs(message=msg))

    cfg = get_config()

    if not config.contrasts or not config.decoding_csp:
        if config.contrasts:
            msg = 'No contrasts specified. '
        else:
            msg = 'No CSP analysis requested. '

        msg += 'Skipping step CSP ...'
        logger.info(**gen_log_kwargs(message=msg))
        return

    subjects = config.get_subjects()
    sessions = config.get_sessions()

    for contrast in config.get_decoding_contrasts():
        parallel, run_func, _ = parallel_func(
            one_subject_decoding,
            n_jobs=config.get_n_jobs()
        )
        logs = parallel(
            run_func(
                cfg=cfg,
                subject=subject,
                session=session,
                contrast=contrast,
            )
            for subject, session in
            itertools.product(subjects, sessions)
        )
        config.save_logs(logs)

    msg = 'Completed Step 4b: CSP'
    logger.info(**gen_log_kwargs(message=msg))


if __name__ == '__main__':
    main()
