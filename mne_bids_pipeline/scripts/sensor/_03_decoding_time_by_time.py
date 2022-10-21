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
from sklearn.model_selection import StratifiedKFold

from ..._config_utils import (
    get_sessions, get_subjects, get_task, get_datatype, get_eeg_reference,
    get_deriv_root, _restrict_analyze_channels, get_decoding_contrasts,
)
from ..._decoding import LogReg
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, save_logs
from ..._parallel import get_parallel_backend, get_n_jobs


def get_input_fnames_time_decoding(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    # TODO: Somehow remove these?
    del kwargs['condition1']
    del kwargs['condition2']
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
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


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_time_decoding)
def run_time_decoding(*, cfg, subject, condition1, condition2, session,
                      in_files):
    msg = f'Contrasting conditions: {condition1} – {condition2}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
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
    with get_parallel_backend(cfg):
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
                n_jobs=cfg.n_jobs,
            )
            cv_scoring_n_jobs = 1
        else:
            estimator = SlidingEstimator(
                clf,
                scoring=cfg.decoding_metric,
                n_jobs=1,
            )
            cv_scoring_n_jobs = cfg.n_jobs

        scores = cross_val_multiscore(
            estimator, X=X, y=y, cv=cv, n_jobs=cv_scoring_n_jobs
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
    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=get_deriv_root(config),
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
        n_jobs=get_n_jobs(config),
        parallel_backend=config.parallel_backend,
        interactive=config.interactive,
        N_JOBS=config.N_JOBS,
    )
    return cfg


def main():
    """Run time-by-time decoding."""
    import config
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
    logs = []
    for subject, session, (cond_1, cond_2) in itertools.product(
        get_subjects(config),
        get_sessions(config),
        get_decoding_contrasts(config)
    ):
        log = run_time_decoding(
            cfg=get_config(config=config),
            subject=subject,
            condition1=cond_1,
            condition2=cond_2,
            session=session,
        )
        logs.append(log)
    save_logs(config=config, logs=logs)


if __name__ == '__main__':
    main()
