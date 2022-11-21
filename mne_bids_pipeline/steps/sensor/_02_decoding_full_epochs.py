"""Decode pairs of conditions based on entire epochs.

Entire epochs are fed into a logistic regression model. Decoding performance
then tells how well the classifier could predict which epoch belongs to
which condition.
"""

###############################################################################
# Let us first import the libraries

import os.path as op
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

import mne
from mne.decoding import Scaler, Vectorizer
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    _restrict_analyze_channels, get_decoding_contrasts
)
from ..._logging import gen_log_kwargs, logger
from ..._decoding import LogReg
from ..._parallel import parallel_func, get_parallel_backend
from ..._run import failsafe_run, save_logs
from ..._report import (
    _open_report, _contrasts_to_names, _plot_full_epochs_decoding_scores,
    _sanitize_cond_tag)


def get_input_fnames_epochs_decoding(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
    condition1: str,
    condition2: str,
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
    get_input_fnames=get_input_fnames_epochs_decoding,
)
def run_epochs_decoding(
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
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(message=msg))
    out_files = dict()
    bids_path = in_files['epochs'].copy()

    epochs = mne.read_epochs(in_files.pop('epochs'))
    _restrict_analyze_channels(epochs, cfg)
    epochs.crop(cfg.decoding_epochs_tmin, cfg.decoding_epochs_tmax)

    # We define the epochs and the labels
    if isinstance(cfg.conditions, dict):
        epochs_conds = [cfg.conditions[condition1],
                        cfg.conditions[condition2]]
        cond_names = [condition1, condition2]
    else:
        epochs_conds = cond_names = [condition1, condition2]
        epochs_conds = [condition1, condition2]

    # We have to use this approach because the conditions could be based on
    # metadata selection, so simply using epochs[conds[0], conds[1]] would
    # not work.
    epochs = mne.concatenate_epochs(
        [epochs[epochs_conds[0]], epochs[epochs_conds[1]]],
        verbose='error')

    n_cond1 = len(epochs[epochs_conds[0]])
    n_cond2 = len(epochs[epochs_conds[1]])

    X = epochs.get_data()
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]

    classification_pipeline = make_pipeline(
        Scaler(scalings='mean'),
        Vectorizer(),  # So we can pass the data to scikit-learn
        LogReg(
            solver='liblinear',  # much faster than the default
            random_state=cfg.random_state,
            n_jobs=1,
        )
    )

    # Now, actually run the classification, and evaluate it via a
    # cross-validation procedure.
    cv = StratifiedKFold(
        shuffle=True,
        random_state=cfg.random_state,
        n_splits=cfg.decoding_n_splits,
    )
    scores = cross_val_score(
        estimator=classification_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1
    )

    # Save the scores
    a_vs_b = f'{cond_names[0]}+{cond_names[1]}'.replace(op.sep, '')
    processing = f'{a_vs_b}+FullEpochs+{cfg.decoding_metric}'
    processing = processing.replace('_', '-').replace('-', '')
    mat_key = f'mat_{processing}'
    tsv_key = f'tsv_{processing}'
    out_files[mat_key] = bids_path.copy().update(
        suffix='decoding', processing=processing, extension='.mat')
    out_files[tsv_key] = out_files[mat_key].copy().update(
        extension='.tsv')
    savemat(out_files[f'mat_{processing}'], {'scores': scores})

    tabular_data = pd.Series(
        {
            'cond_1': cond_names[0],
            'cond_2': cond_names[1],
            'mean_crossval_score': scores.mean(axis=0),
            'metric': cfg.decoding_metric
        }
    )
    tabular_data = pd.DataFrame(tabular_data).T
    tabular_data.to_csv(out_files[tsv_key], sep='\t', index=False)

    # Report
    with _open_report(
            cfg=cfg,
            exec_params=exec_params,
            subject=subject,
            session=session) as report:
        msg = 'Adding full-epochs decoding results to the report.'
        logger.info(**gen_log_kwargs(message=msg))

        all_decoding_scores = []
        all_contrasts = []
        for contrast in cfg.contrasts:
            cond_1, cond_2 = contrast
            a_vs_b = f'{cond_1}+{cond_2}'.replace(op.sep, '')
            processing = f'{a_vs_b}+FullEpochs+{cfg.decoding_metric}'
            processing = processing.replace('_', '-').replace('-', '')
            fname_decoding = bids_path.copy().update(
                processing=processing,
                suffix='decoding',
                extension='.mat'
            )
            if not fname_decoding.fpath.is_file():  # not done yet
                continue
            decoding_data = loadmat(fname_decoding)
            all_decoding_scores.append(
                np.atleast_1d(decoding_data['scores'].squeeze())
            )
            all_contrasts.append(contrast)
            del fname_decoding, processing, a_vs_b, decoding_data

        fig, caption = _plot_full_epochs_decoding_scores(
            contrast_names=_contrasts_to_names(all_contrasts),
            scores=all_decoding_scores,
            metric=cfg.decoding_metric,
        )
        report.add_figure(
            fig=fig,
            title='Full-epochs decoding',
            caption=caption,
            section='Decoding: full-epochs',
            tags=(
                'epochs',
                'contrast',
                'decoding',
                *[f'{_sanitize_cond_tag(cond_1)}–'
                  f'{_sanitize_cond_tag(cond_2)}'
                  for cond_1, cond_2 in cfg.contrasts]
            ),
            replace=True,
        )
        # close figure to save memory
        plt.close(fig)

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
        decoding_epochs_tmin=config.decoding_epochs_tmin,
        decoding_epochs_tmax=config.decoding_epochs_tmax,
        decoding_n_splits=config.decoding_n_splits,
        random_state=config.random_state,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config)
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

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_epochs_decoding, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                exec_params=config.exec_params,
                subject=subject,
                condition1=cond_1,
                condition2=cond_2,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
            for (cond_1, cond_2) in get_decoding_contrasts(config)
        )
    save_logs(config=config, logs=logs)
