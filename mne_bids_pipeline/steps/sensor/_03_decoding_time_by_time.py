"""Decode time-by-time using a "sliding" estimator.

A sliding estimator fits a separate logistic regression model for every time
point. The end result is an averaging effect across sensors.

This approach is different from the one taken in the decoding script for
entire epochs. Here, the classifier is traines on the entire epoch, and hence
can learn about the entire time course of the signal.
"""

###############################################################################
# Let us first import the libraries

import os.path as op
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

import mne
from mne.decoding import (
    GeneralizingEstimator, SlidingEstimator, cross_val_multiscore
)

from mne_bids import BIDSPath

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    _restrict_analyze_channels, get_decoding_contrasts,
)
from ..._decoding import LogReg
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, save_logs
from ..._parallel import get_parallel_backend, get_parallel_backend_name
from ..._report import (
    _open_report, _plot_decoding_time_generalization, _sanitize_cond_tag,
    _plot_time_by_time_decoding_scores,
)


def get_input_fnames_time_decoding(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    condition1: str,
    condition2: str,
) -> dict:
    # TODO: Shouldn't this at least use the PTP-rejected epochs if available?
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
    get_input_fnames=get_input_fnames_time_decoding,
)
def run_time_decoding(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    condition1: str,
    condition2: str,
    in_files: dict,
) -> dict:
    import matplotlib.pyplot as plt
    if cfg.decoding_time_generalization:
        kind = 'time generalization'
    else:
        kind = 'sliding estimator'
    msg = f'Contrasting conditions ({kind}): {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(message=msg))
    out_files = dict()
    bids_path = in_files['epochs'].copy()

    epochs = mne.read_epochs(in_files.pop('epochs'))
    _restrict_analyze_channels(epochs, cfg)

    # We define the epochs and the labels
    if isinstance(cfg.conditions, dict):
        epochs_conds = [cfg.conditions[condition1],
                        cfg.conditions[condition2]]
        cond_names = [condition1, condition2]
    else:
        epochs_conds = cond_names = [condition1, condition2]
        epochs_conds = [condition1, condition2]
    epoch_counts = dict()
    for contrast in cfg.contrasts:
        for cond in contrast:
            if cond not in epoch_counts:
                epoch_counts[cond] = len(epochs[cond])

    # We have to use this approach because the conditions could be based on
    # metadata selection, so simply using epochs[conds[0], conds[1]] would
    # not work.
    epochs = mne.concatenate_epochs([epochs[epochs_conds[0]],
                                     epochs[epochs_conds[1]]])
    n_cond1 = len(epochs[epochs_conds[0]])
    n_cond2 = len(epochs[epochs_conds[1]])

    decim = cfg.decoding_time_generalization_decim
    if cfg.decoding_time_generalization and decim > 1:
        epochs.decimate(decim, verbose='error')

    X = epochs.get_data()
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]
    # ProgressBar does not work on dask, so only enable it if not using dask
    verbose = get_parallel_backend_name(exec_params=exec_params) != "dask"
    with get_parallel_backend(exec_params):
        clf = make_pipeline(
            StandardScaler(),
            LogReg(
                solver='liblinear',  # much faster than the default
                random_state=cfg.random_state,
                n_jobs=1,
            )
        )
        cv = StratifiedKFold(
            shuffle=True,
            random_state=cfg.random_state,
            n_splits=cfg.decoding_n_splits,
        )

        if cfg.decoding_time_generalization:
            estimator = GeneralizingEstimator(
                clf,
                scoring=cfg.decoding_metric,
                n_jobs=exec_params.N_JOBS,
            )
            cv_scoring_n_jobs = 1
        else:
            estimator = SlidingEstimator(
                clf,
                scoring=cfg.decoding_metric,
                n_jobs=1,
            )
            cv_scoring_n_jobs = exec_params.N_JOBS

        scores = cross_val_multiscore(
            estimator, X=X, y=y, cv=cv, n_jobs=cv_scoring_n_jobs,
            verbose=verbose,  # ensure ProgressBar is shown (can be slow)
        )

        # let's save the scores now
        a_vs_b = f'{cond_names[0]}+{cond_names[1]}'.replace(op.sep, '')
        processing = f'{a_vs_b}+TimeByTime+{cfg.decoding_metric}'
        processing = processing.replace('_', '-').replace('-', '')
        mat_key = f'mat_{processing}'
        out_files[mat_key] = bids_path.copy().update(
            suffix='decoding', processing=processing, extension='.mat')
        savemat(
            out_files[mat_key],
            {
                'scores': scores,
                'times': epochs.times,
                'decim': decim,
            }
        )

        if cfg.decoding_time_generalization:
            # Only store the mean scores for the diagonal in the TSV file –
            # we still have all time generalization results in the MAT file
            # we just saved.
            mean_crossval_score = np.diag(scores.mean(axis=0))
        else:
            mean_crossval_score = scores.mean(axis=0)

        out_files[f'tsv_{processing}'] = out_files[mat_key].copy().update(
            extension='.tsv')
        tabular_data = pd.DataFrame(
            dict(cond_1=[cond_names[0]] * len(epochs.times),
                 cond_2=[cond_names[1]] * len(epochs.times),
                 time=epochs.times,
                 mean_crossval_score=mean_crossval_score,
                 metric=[cfg.decoding_metric] * len(epochs.times))
        )
        tabular_data.to_csv(
            out_files[f'tsv_{processing}'], sep='\t', index=False)

    # Report
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        msg = 'Adding time-by-time decoding results to the report.'
        logger.info(**gen_log_kwargs(message=msg))

        section = 'Decoding: time-by-time'
        for contrast in cfg.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            tags = (
                'epochs',
                'contrast',
                'decoding',
                f"{_sanitize_cond_tag(contrast[0])}–"
                f"{_sanitize_cond_tag(contrast[1])}"
            )

            processing = f'{a_vs_b}+TimeByTime+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding = bids_path.copy().update(
                processing=processing,
                suffix='decoding',
                extension='.mat'
            )
            if not fname_decoding.fpath.is_file():
                continue
            decoding_data = loadmat(fname_decoding)
            del fname_decoding, processing, a_vs_b

            fig = _plot_time_by_time_decoding_scores(
                times=decoding_data['times'].ravel(),
                cross_val_scores=decoding_data['scores'],
                metric=cfg.decoding_metric,
                time_generalization=cfg.decoding_time_generalization,
                decim=decoding_data['decim'].item(),
            )
            caption = (
                f'Time-by-time decoding: '
                f'{epoch_counts[cond_1]} × {cond_1} vs. '
                f'{epoch_counts[cond_2]} × {cond_2}'
            )
            title = f'Decoding over time: {cond_1} vs. {cond_2}'
            report.add_figure(
                fig=fig,
                title=title,
                caption=caption,
                section=section,
                tags=tags,
                replace=True,
            )
            plt.close(fig)

            if cfg.decoding_time_generalization:
                fig = _plot_decoding_time_generalization(
                    decoding_data=decoding_data,
                    metric=cfg.decoding_metric,
                    kind='single-subject'
                )
                caption = (
                    'Time generalization (generalization across time, GAT): '
                    'each classifier is trained on each time point, and '
                    'tested on all other time points.'
                )
                title = f'Time generalization: {cond_1} vs. {cond_2}'
                report.add_figure(
                    fig=fig,
                    title=title,
                    caption=caption,
                    section=section,
                    tags=tags,
                    replace=True,
                )
                plt.close(fig)

            del decoding_data, cond_1, cond_2, caption

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.deriv_root,
        conditions=config.conditions,
        contrasts=get_decoding_contrasts(config),
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_n_splits=config.decoding_n_splits,
        decoding_time_generalization=config.decoding_time_generalization,
        decoding_time_generalization_decim=config.decoding_time_generalization_decim,  # noqa: E501
        random_state=config.random_state,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run time-by-time decoding."""
    if not config.contrasts:
        msg = 'No contrasts specified; not performing decoding.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    if not config.decode:
        msg = 'No decoding requested by user.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    # Here we go parallel inside the :class:`mne.decoding.SlidingEstimator`
    # so we don't dispatch manually to multiple jobs.
    logs = [
        run_time_decoding(
            cfg=get_config(
                config=config,
            ),
            exec_params=config.exec_params,
            subject=subject,
            condition1=cond_1,
            condition2=cond_2,
            session=session,
        )
        for subject in get_subjects(config)
        for session in get_sessions(config)
        for cond_1, cond_2 in get_decoding_contrasts(config)
    ]
    save_logs(config=config, logs=logs)
