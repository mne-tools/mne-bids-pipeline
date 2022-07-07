"""
=================================================
Time-by-time decoding using a "sliding" estimator
=================================================

A sliding estimator fits a separate logistic regression model for every time
point. The end result is an averaging effect across sensors.

This approach is different from the one taken in the decoding script for
entire epochs. Here, the classifier is traines on the entire epoch, and hence
can learn about the entire time course of the signal.
"""

###############################################################################
# Let us first import the libraries

import os.path as op
import logging
from typing import Optional
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import savemat

import mne
from mne.decoding import (
    GeneralizingEstimator, SlidingEstimator, cross_val_multiscore
)

from mne_bids import BIDSPath

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import config
from config import gen_log_kwargs, on_error, failsafe_run, parallel_func


logger = logging.getLogger('mne-bids-pipeline')


class LogReg(LogisticRegression):
    """Hack to avoid a warning with n_jobs != 1 when using dask
    """
    def fit(self, *args, **kwargs):
        from joblib import parallel_backend
        with parallel_backend("loky"):
            return super().fit(*args, **kwargs)


@failsafe_run(on_error=on_error, script_path=__file__)
def run_time_decoding(*, cfg, subject, condition1, condition2, session=None):
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

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
    epochs = mne.concatenate_epochs([epochs[epochs_conds[0]],
                                     epochs[epochs_conds[1]]])
    n_cond1 = len(epochs[epochs_conds[0]])
    n_cond2 = len(epochs[epochs_conds[1]])

    X = epochs.get_data()
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]
    clf = make_pipeline(
        StandardScaler(),
        LogReg(
            solver='liblinear',  # much faster than the default
            random_state=cfg.random_state,
            n_jobs=1,  # higher values don't make sense for liblinear
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
            n_jobs=1
        )
    else:
        estimator = SlidingEstimator(
            clf,
            scoring=cfg.decoding_metric,
            n_jobs=1
        )

    scores = cross_val_multiscore(
        estimator, X=X, y=y, cv=cv, n_jobs=None
    )

    # let's save the scores now
    a_vs_b = f'{cond_names[0]}+{cond_names[1]}'.replace(op.sep, '')
    processing = f'{a_vs_b}+TimeByTime+{cfg.decoding_metric}'
    processing = processing.replace('_', '-').replace('-', '')

    fname_mat = (
        fname_epochs
        .copy()
        .update(suffix='decoding',
                processing=processing,
                extension='.mat')
    )
    savemat(fname_mat, {'scores': scores, 'times': epochs.times})

    if cfg.decoding_time_generalization:
        # Only store the mean scores for the diagonal in the TSV file –
        # we still have all time generalization results in the MAT file
        # we just saved.
        mean_crossval_score = np.diag(scores.mean(axis=0))
    else:
        mean_crossval_score = scores.mean(axis=0)

    fname_tsv = fname_mat.copy().update(extension='.tsv')
    tabular_data = pd.DataFrame(
        dict(cond_1=[cond_names[0]] * len(epochs.times),
             cond_2=[cond_names[1]] * len(epochs.times),
             time=epochs.times,
             mean_crossval_score=mean_crossval_score,
             metric=[cfg.decoding_metric] * len(epochs.times))
    )
    tabular_data.to_csv(fname_tsv, sep='\t', index=False)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        conditions=config.conditions,
        contrasts=config.get_decoding_contrasts(),
        decode=config.decode,
        decoding_metric=config.decoding_metric,
        decoding_n_splits=config.decoding_n_splits,
        decoding_time_generalization=config.decoding_time_generalization,
        random_state=config.random_state,
        analyze_channels=config.analyze_channels,
        ch_types=config.ch_types,
        eeg_reference=config.get_eeg_reference(),
        n_jobs=config.get_n_jobs(),
    )
    return cfg


def main():
    """Run time-by-time decoding."""
    if not config.contrasts:
        msg = 'No contrasts specified; not performing decoding.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    if not config.decode:
        msg = 'No decoding requested by user.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    # Extremely ugly (?) hack to get things to parallelize more nicely without
    # a resource over-subcription when processing fewer than n_jobs
    # datasets (participants x sessions x contrasts).
    # Only tested with loky so far!
    inner_max_num_threads = config.get_n_jobs()

    with config.get_parallel_backend(
        inner_max_num_threads=inner_max_num_threads
    ):
        parallel, run_func = parallel_func(run_time_decoding)
        logs = parallel(
            run_func(
                cfg=get_config(subject, session),
                subject=subject,
                session=session,
                condition1=contrast[0],
                condition2=contrast[1]
            )
            for subject, session, contrast in itertools.product(
                config.get_subjects(),
                config.get_sessions(),
                config.get_decoding_contrasts()
            )
        )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
