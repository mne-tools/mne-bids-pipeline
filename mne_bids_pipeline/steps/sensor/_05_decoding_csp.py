"""
Decoding based on common spatial patterns (CSP).
"""

import os.path as op
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import matplotlib.transforms
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne_bids import BIDSPath
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    _restrict_analyze_channels, get_decoding_contrasts,
)
from ..._decoding import LogReg, _handle_csp_args
from ..._logging import logger, gen_log_kwargs
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run, save_logs
from ..._report import (
    _open_report, _sanitize_cond_tag, _plot_full_epochs_decoding_scores,
    _imshow_tf,
)


def _prepare_labels(
    *,
    epochs: mne.BaseEpochs,
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
    epochs: mne.BaseEpochs,
    contrast: Tuple[str, str],
    cfg,
    fmin: float,
    fmax: float
) -> Tuple[mne.BaseEpochs, np.ndarray]:
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


def get_input_fnames_csp(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    contrast: Tuple[str],
) -> dict:
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
    get_input_fnames=get_input_fnames_csp
)
def one_subject_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str,
    contrast: Tuple[str, str],
    in_files: Dict[str, BIDSPath]
) -> dict:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.
    """
    import matplotlib.pyplot as plt
    condition1, condition2 = contrast
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(msg))

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
    logger.info(**gen_log_kwargs(msg))
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
    freq_name_to_bins_map = _handle_csp_args(
        cfg.decoding_csp_times, cfg.decoding_csp_freqs, cfg.decoding_metric)
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
        logger.info(**gen_log_kwargs(msg))

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
        logger.info(**gen_log_kwargs(msg))

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
    out_files = {'csp-excel': fname_results}

    # Report
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        msg = 'Adding CSP decoding results to the report.'
        logger.info(**gen_log_kwargs(message=msg))
        section = 'Decoding: CSP'
        freq_name_to_bins_map = _handle_csp_args(
            cfg.decoding_csp_times,
            cfg.decoding_csp_freqs,
            cfg.decoding_metric,
        )
        all_csp_tf_results = dict()
        for contrast in cfg.decoding_contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            tags = (
                'epochs',
                'contrast',
                'decoding',
                'csp',
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}"
            )
            processing = f'{a_vs_b}+CSP+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding = bids_path.copy().update(
                processing=processing,
                suffix='decoding',
                extension='.xlsx'
            )
            if not fname_decoding.fpath.is_file():
                continue  # not done yet
            csp_freq_results = pd.read_excel(
                fname_decoding,
                sheet_name='CSP Frequency'
            )
            csp_freq_results['scores'] = csp_freq_results['scores'].apply(
                lambda x: np.array(x[1:-1].split(), float))
            csp_tf_results = pd.read_excel(
                fname_decoding,
                sheet_name='CSP Time-Frequency'
            )
            csp_tf_results['scores'] = csp_tf_results['scores'].apply(
                lambda x: np.array(x[1:-1].split(), float))
            all_csp_tf_results[contrast] = csp_tf_results
            del csp_tf_results

            all_decoding_scores = list()
            contrast_names = list()
            for freq_range_name, freq_bins in freq_name_to_bins_map.items():
                results = csp_freq_results.loc[
                    csp_freq_results['freq_range_name'] == freq_range_name
                ]
                results.reset_index(drop=True, inplace=True)
                assert len(results['scores']) == len(freq_bins)
                for bi, freq_bin in enumerate(freq_bins):
                    all_decoding_scores.append(results['scores'][bi])
                    f_min = float(freq_bin[0])
                    f_max = float(freq_bin[1])
                    contrast_names.append(
                        f'{freq_range_name}\n'
                        f'({f_min:0.1f}-{f_max:0.1f} Hz)'
                    )
            fig, caption = _plot_full_epochs_decoding_scores(
                contrast_names=contrast_names,
                scores=all_decoding_scores,
                metric=cfg.decoding_metric,
            )
            title = f'CSP decoding: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                caption=caption,
                tags=tags,
                replace=True,
            )
            # close figure to save memory
            plt.close(fig)
            del fig, caption, title

        # Now, plot decoding scores across time-frequency bins.
        for contrast in cfg.decoding_contrasts:
            if contrast not in all_csp_tf_results:
                continue
            cond_1, cond_2 = contrast
            tags = (
                'epochs',
                'contrast',
                'decoding',
                'csp',
                f"{_sanitize_cond_tag(cond_1)}–{_sanitize_cond_tag(cond_2)}",
            )
            results = all_csp_tf_results[contrast]
            mean_crossval_scores = list()
            tmin, tmax, fmin, fmax = list(), list(), list(), list()
            mean_crossval_scores.extend(
                results['mean_crossval_score'].ravel())
            tmin.extend(results['t_min'].ravel())
            tmax.extend(results['t_max'].ravel())
            fmin.extend(results['f_min'].ravel())
            fmax.extend(results['f_max'].ravel())
            mean_crossval_scores = np.array(mean_crossval_scores, float)
            fig, ax = plt.subplots(constrained_layout=True)
            # XXX Add support for more metrics
            assert cfg.decoding_metric == 'roc_auc'
            metric = 'ROC AUC'
            vmax = max(
                np.abs(mean_crossval_scores.min() - 0.5),
                np.abs(mean_crossval_scores.max() - 0.5)
            ) + 0.5
            vmin = 0.5 - (vmax - 0.5)
            img = _imshow_tf(
                mean_crossval_scores, ax,
                tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                vmin=vmin, vmax=vmax)
            offset = matplotlib.transforms.offset_copy(
                ax.transData, fig, 6, 0, units='points')
            for freq_range_name, bins in freq_name_to_bins_map.items():
                ax.text(tmin[0],
                        0.5 * bins[0][0] + 0.5 * bins[-1][1],
                        freq_range_name, transform=offset,
                        ha='left', va='center', rotation=90)
            ax.set_xlim([np.min(tmin), np.max(tmax)])
            ax.set_ylim([np.min(fmin), np.max(fmax)])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            cbar = fig.colorbar(
                ax=ax, shrink=0.75, orientation='vertical', mappable=img)
            cbar.set_label(f'Mean decoding score ({metric})')
            title = f'CSP TF decoding: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                section=section,
                tags=tags,
                replace=True,
            )

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str]
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        # Data parameters
        datatype=get_datatype(config),
        deriv_root=config.deriv_root,
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
        decoding_contrasts=get_decoding_contrasts(config),
        n_boot=config.n_boot,
        random_state=config.random_state,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run all subjects decoding in parallel."""
    if not config.contrasts or not config.decoding_csp:
        if not config.contrasts:
            msg = 'No contrasts specified. '
        else:
            msg = 'No CSP analysis requested. '

        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            one_subject_decoding, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                    session=session
                ),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                contrast=contrast,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for contrast in get_decoding_contrasts(config)
        )
        save_logs(logs=logs, config=config)
