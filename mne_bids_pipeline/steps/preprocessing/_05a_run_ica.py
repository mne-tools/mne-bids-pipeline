"""Run Independent Component Analysis (ICA) for artifact correction.

This fits ICA on epoched data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.

Before performing ICA, we reject epochs based on peak-to-peak amplitude above
the 'ica_reject' to filter massive non-biological artifacts.

To actually remove designated ICA components from your data, you will have to
run 05a-apply_ica.py.
"""

from typing import List, Optional, Iterable, Tuple, Literal
from types import SimpleNamespace

import pandas as pd
import numpy as np

import mne
from mne.report import Report
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne_bids import BIDSPath

from ..._config_utils import (
    get_sessions, get_runs, get_subjects, get_task, get_datatype,
    get_eeg_reference,
)
from ..._import_data import make_epochs, annotations_to_events
from ..._logging import gen_log_kwargs, logger
from ..._parallel import parallel_func, get_parallel_backend
from ..._reject import _get_reject
from ..._report import _agg_backend
from ..._run import failsafe_run, _update_for_splits, save_logs


def filter_for_ica(
    *,
    cfg,
    raw: mne.io.BaseRaw,
    subject: str,
    session: Optional[str],
    run: Optional[str] = None
) -> None:
    """Apply a high-pass filter if needed."""
    if cfg.ica_l_freq is None:
        msg = (f'Not applying high-pass filter (data is already filtered, '
               f'cutoff: {raw.info["highpass"]} Hz).')
        logger.info(**gen_log_kwargs(message=msg))
    else:
        msg = f'Applying high-pass filter with {cfg.ica_l_freq} Hz cutoff …'
        logger.info(**gen_log_kwargs(message=msg))
        raw.filter(l_freq=cfg.ica_l_freq, h_freq=None, n_jobs=1)


def fit_ica(
    *,
    cfg,
    epochs: mne.BaseEpochs,
    subject: str,
    session: Optional[str],
) -> mne.preprocessing.ICA:
    algorithm = cfg.ica_algorithm
    fit_params = None

    if algorithm == 'picard':
        fit_params = dict(fastica_it=5)
    elif algorithm == 'extended_infomax':
        algorithm = 'infomax'
        fit_params = dict(extended=True)

    ica = ICA(method=algorithm, random_state=cfg.random_state,
              n_components=cfg.ica_n_components, fit_params=fit_params,
              max_iter=cfg.ica_max_iterations)

    ica.fit(epochs, decim=cfg.ica_decim)

    explained_var = (ica.pca_explained_variance_[:ica.n_components_].sum() /
                     ica.pca_explained_variance_.sum())
    msg = (f'Fit {ica.n_components_} components (explaining '
           f'{round(explained_var * 100, 1)}% of the variance) in '
           f'{ica.n_iter_} iterations.')
    logger.info(**gen_log_kwargs(message=msg))
    return ica


def make_ecg_epochs(
    *,
    cfg,
    raw_path: BIDSPath,
    subject: str,
    session: Optional[str],
    run: Optional[str] = None,
    n_runs: int
) -> Optional[mne.BaseEpochs]:
    # ECG either needs an ecg channel, or avg of the mags (i.e. MEG data)
    raw = mne.io.read_raw(raw_path, preload=False)

    if ('ecg' in raw.get_channel_types() or 'meg' in cfg.ch_types or
            'mag' in cfg.ch_types):
        msg = 'Creating ECG epochs …'
        logger.info(**gen_log_kwargs(message=msg))

        # We want to extract a total of 5 min of data for ECG epochs generation
        # (across all runs)
        total_ecg_dur = 5 * 60
        ecg_dur_per_run = total_ecg_dur / n_runs
        t_mid = (raw.times[-1] + raw.times[0]) / 2
        raw = raw.crop(
            tmin=max(t_mid - 1/2 * ecg_dur_per_run, 0),
            tmax=min(t_mid + 1/2 * ecg_dur_per_run, raw.times[-1])
        ).load_data()

        ecg_epochs = create_ecg_epochs(raw,
                                       baseline=(None, -0.2),
                                       tmin=-0.5, tmax=0.5)
        del raw  # Free memory

        if len(ecg_epochs) == 0:
            msg = ('No ECG events could be found. Not running ECG artifact '
                   'detection.')
            logger.info(**gen_log_kwargs(message=msg))
            ecg_epochs = None
    else:
        msg = ('No ECG or magnetometer channels are present. Cannot '
               'automate artifact detection for ECG')
        logger.info(**gen_log_kwargs(message=msg))
        ecg_epochs = None

    return ecg_epochs


def make_eog_epochs(
    *,
    raw: mne.io.BaseRaw,
    eog_channels: Optional[Iterable[str]],
    subject: str,
    session: Optional[str],
    run: Optional[str] = None
) -> Optional[mne.Epochs]:
    """Create EOG epochs. No rejection thresholds will be applied.
    """
    if eog_channels:
        ch_names = eog_channels
        assert all([ch_name in raw.ch_names
                    for ch_name in ch_names])
    else:
        ch_idx = mne.pick_types(raw.info, meg=False, eog=True)
        ch_names = [raw.ch_names[i] for i in ch_idx]
        del ch_idx

    if ch_names:
        msg = 'Creating EOG epochs …'
        logger.info(**gen_log_kwargs(message=msg))

        eog_epochs = create_eog_epochs(raw, ch_name=ch_names,
                                       baseline=(None, -0.2))

        if len(eog_epochs) == 0:
            msg = ('No EOG events could be found. Not running EOG artifact '
                   'detection.')
            logger.warning(**gen_log_kwargs(message=msg))
            eog_epochs = None
    else:
        msg = ('No EOG channel is present. Cannot automate IC detection '
               'for EOG')
        logger.info(**gen_log_kwargs(message=msg))
        eog_epochs = None

    return eog_epochs


def detect_bad_components(
    *,
    cfg,
    which: Literal['eog', 'ecg'],
    epochs: mne.BaseEpochs,
    ica: mne.preprocessing.ICA,
    ch_names: Optional[List[str]],
    subject: str,
    session: str
) -> Tuple[List[int], np.ndarray]:
    artifact = which.upper()
    msg = f'Performing automated {artifact} artifact detection …'
    logger.info(**gen_log_kwargs(message=msg))

    if which == 'eog':
        inds, scores = ica.find_bads_eog(
            epochs,
            threshold=cfg.ica_eog_threshold,
            ch_name=ch_names,
        )
    else:
        inds, scores = ica.find_bads_ecg(
            epochs, method='ctps',
            threshold=cfg.ica_ctps_ecg_threshold,
            ch_name=ch_names,
        )

    if not inds:
        adjust_setting = ('ica_eog_threshold' if which == 'eog'
                          else 'ica_ctps_ecg_threshold')
        warn = (f'No {artifact}-related ICs detected, this is highly '
                f'suspicious. A manual check is suggested. You may wish to '
                f'lower "{adjust_setting}".')
        logger.warning(**gen_log_kwargs(message=warn))
    else:
        msg = (f'Detected {len(inds)} {artifact}-related ICs in '
               f'{len(epochs)} {artifact} epochs.')
        logger.info(**gen_log_kwargs(message=msg))

    return inds, scores


def get_input_fnames_run_ica(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
) -> dict:
    bids_basename = BIDSPath(subject=subject,
                             session=session,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             recording=cfg.rec,
                             space=cfg.space,
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             check=False)
    in_files = dict()
    for run in cfg.runs:
        key = f'raw_run-{run}'
        in_files[key] = bids_basename.copy().update(
            run=run, processing='filt', suffix='raw')
        _update_for_splits(in_files, key, single=True)
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_run_ica,
)
def run_ica(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: Optional[str],
    in_files: dict,
) -> dict:
    """Run ICA."""
    raw_fnames = [in_files.pop(f'raw_run-{run}') for run in cfg.runs]
    bids_basename = raw_fnames[0].copy().update(
        processing=None, split=None, run=None)
    out_files = dict()
    out_files['ica'] = bids_basename.copy().update(
        suffix='ica', extension='.fif')
    out_files['components'] = bids_basename.copy().update(
        processing='ica', suffix='components', extension='.tsv')
    out_files['report'] = bids_basename.copy().update(
        processing='ica+components', suffix='report', extension='.html')
    del bids_basename

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.

    # Generate a unique event name -> event code mapping that can be used
    # across all runs.
    event_name_to_code_map = annotations_to_events(raw_paths=raw_fnames)

    # Now, generate epochs from each individual run
    eog_epochs_all_runs = None
    ecg_epochs_all_runs = None

    for idx, (run, raw_fname) in enumerate(
        zip(cfg.runs, raw_fnames)
    ):
        msg = f'Loading filtered raw data from {raw_fname.basename}'
        logger.info(**gen_log_kwargs(message=msg))

        # ECG epochs
        ecg_epochs = make_ecg_epochs(
            cfg=cfg, raw_path=raw_fname, subject=subject, session=session,
            run=run, n_runs=len(cfg.runs)
        )
        if ecg_epochs is not None:
            if idx == 0:
                ecg_epochs_all_runs = ecg_epochs
            else:
                ecg_epochs_all_runs = mne.concatenate_epochs(
                    [ecg_epochs_all_runs, ecg_epochs], on_mismatch='warn'
                )

            del ecg_epochs

        # EOG epochs
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        eog_epochs = make_eog_epochs(
            raw=raw, eog_channels=cfg.eog_channels, subject=subject,
            session=session, run=run
        )
        if eog_epochs is not None:
            if idx == 0:
                eog_epochs_all_runs = eog_epochs
            else:
                eog_epochs_all_runs = mne.concatenate_epochs(
                    [eog_epochs_all_runs, eog_epochs], on_mismatch='warn'
                )

            del eog_epochs

        # Produce high-pass filtered version of the data for ICA.
        # Sanity check – make sure we're using the correct data!
        if cfg.raw_resample_sfreq is not None:
            assert np.allclose(raw.info['sfreq'], cfg.raw_resample_sfreq)
        if cfg.l_freq is not None:
            assert np.allclose(raw.info['highpass'], cfg.l_freq)

        filter_for_ica(cfg=cfg, raw=raw, subject=subject, session=session,
                       run=run)

        # Only keep the subset of the mapping that applies to the current run
        event_id = event_name_to_code_map.copy()
        for event_name in event_id.copy().keys():
            if event_name not in raw.annotations.description:
                del event_id[event_name]

        msg = 'Creating task-related epochs …'
        logger.info(**gen_log_kwargs(message=msg))
        epochs = make_epochs(
            subject=subject,
            session=session,
            task=cfg.task,
            conditions=cfg.conditions,
            raw=raw,
            event_id=event_id,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            metadata_tmin=cfg.epochs_metadata_tmin,
            metadata_tmax=cfg.epochs_metadata_tmax,
            metadata_keep_first=cfg.epochs_metadata_keep_first,
            metadata_keep_last=cfg.epochs_metadata_keep_last,
            metadata_query=cfg.epochs_metadata_query,
            event_repeated=cfg.event_repeated,
            epochs_decim=cfg.epochs_decim,
            task_is_rest=cfg.task_is_rest,
            rest_epochs_duration=cfg.rest_epochs_duration,
            rest_epochs_overlap=cfg.rest_epochs_overlap,
        )

        epochs.load_data()  # Remove reference to raw
        del raw  # free memory

        if idx == 0:
            epochs_all_runs = epochs
        else:
            epochs_all_runs = mne.concatenate_epochs(
                [epochs_all_runs, epochs], on_mismatch='warn'
            )

        del epochs

    # Clean up namespace
    epochs = epochs_all_runs
    epochs_ecg = ecg_epochs_all_runs
    epochs_eog = eog_epochs_all_runs

    del epochs_all_runs, eog_epochs_all_runs, ecg_epochs_all_runs

    # Set an EEG reference
    if 'eeg' in cfg.ch_types:
        projection = True if cfg.eeg_reference == 'average' else False
        epochs.set_eeg_reference(cfg.eeg_reference, projection=projection)

    # Reject epochs based on peak-to-peak rejection thresholds
    ica_reject = _get_reject(
        subject=subject,
        session=session,
        reject=cfg.ica_reject,
        ch_types=cfg.ch_types,
        param='ica_reject',
    )

    msg = f'Using PTP rejection thresholds: {ica_reject}'
    logger.info(**gen_log_kwargs(message=msg))

    epochs.drop_bad(reject=ica_reject)
    if epochs_eog is not None:
        epochs_eog.drop_bad(reject=ica_reject)
    if epochs_ecg is not None:
        epochs_ecg.drop_bad(reject=ica_reject)

    # Now actually perform ICA.
    msg = 'Calculating ICA solution.'
    logger.info(**gen_log_kwargs(message=msg))
    ica = fit_ica(cfg=cfg, epochs=epochs, subject=subject, session=session)

    # Start a report
    title = f'ICA – sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    # ECG and EOG component detection
    if epochs_ecg:
        ecg_ics, ecg_scores = detect_bad_components(
            cfg=cfg,
            which='ecg',
            epochs=epochs_ecg,
            ica=ica,
            ch_names=None,  # we currently don't allow for custom channels
            subject=subject,
            session=session
        )
    else:
        ecg_ics = ecg_scores = []

    if epochs_eog:
        eog_ics, eog_scores = detect_bad_components(
            cfg=cfg,
            which='eog',
            epochs=epochs_eog,
            ica=ica,
            ch_names=cfg.eog_channels,
            subject=subject,
            session=session
        )
    else:
        eog_ics = eog_scores = []

    # Save ICA to disk.
    # We also store the automatically identified ECG- and EOG-related ICs.
    msg = 'Saving ICA solution and detected artifacts to disk.'
    logger.info(**gen_log_kwargs(message=msg))
    ica.exclude = sorted(set(ecg_ics + eog_ics))
    ica.save(out_files['ica'], overwrite=True)
    _update_for_splits(out_files, 'ica')

    # Create TSV.
    tsv_data = pd.DataFrame(
        dict(component=list(range(ica.n_components_)),
             type=['ica'] * ica.n_components_,
             description=['Independent Component'] * ica.n_components_,
             status=['good'] * ica.n_components_,
             status_description=['n/a'] * ica.n_components_))

    for component in ecg_ics:
        row_idx = tsv_data['component'] == component
        tsv_data.loc[row_idx, 'status'] = 'bad'
        tsv_data.loc[row_idx,
                     'status_description'] = 'Auto-detected ECG artifact'

    for component in eog_ics:
        row_idx = tsv_data['component'] == component
        tsv_data.loc[row_idx, 'status'] = 'bad'
        tsv_data.loc[row_idx,
                     'status_description'] = 'Auto-detected EOG artifact'

    tsv_data.to_csv(out_files['components'], sep='\t', index=False)

    # Lastly, add info about the epochs used for the ICA fit, and plot all ICs
    # for manual inspection.
    msg = 'Adding diagnostic plots for all ICA components to the HTML report …'
    logger.info(**gen_log_kwargs(message=msg))

    report = Report(info_fname=epochs, title=title, verbose=False)
    ecg_evoked = None if epochs_ecg is None else epochs_ecg.average()
    eog_evoked = None if epochs_eog is None else epochs_eog.average()
    ecg_scores = None if len(ecg_scores) == 0 else ecg_scores
    eog_scores = None if len(eog_scores) == 0 else eog_scores

    with _agg_backend():
        report.add_epochs(
            epochs=epochs,
            title='Epochs used for ICA fitting',
            drop_log_ignore=(),
            replace=True,
        )
        report.add_ica(
            ica=ica,
            title='ICA cleaning',
            inst=epochs,
            ecg_evoked=ecg_evoked,
            eog_evoked=eog_evoked,
            ecg_scores=ecg_scores,
            eog_scores=eog_scores,
            replace=True,
            n_jobs=1,  # avoid automatic parallelization
        )

    msg = (f"ICA completed. Please carefully review the extracted ICs in the "
           f"report {out_files['report'].basename}, and mark all components "
           f"you wish to reject as 'bad' in "
           f"{out_files['components'].basename}")
    logger.info(**gen_log_kwargs(message=msg))

    report.save(
        out_files['report'],
        overwrite=True,
        open_browser=exec_params.interactive,
    )

    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    *,
    config: SimpleNamespace,
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        conditions=config.conditions,
        task=get_task(config),
        task_is_rest=config.task_is_rest,
        datatype=get_datatype(config),
        runs=get_runs(config=config, subject=subject),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.deriv_root,
        ica_l_freq=config.ica_l_freq,
        ica_algorithm=config.ica_algorithm,
        ica_n_components=config.ica_n_components,
        ica_max_iterations=config.ica_max_iterations,
        ica_decim=config.ica_decim,
        ica_reject=config.ica_reject,
        ica_eog_threshold=config.ica_eog_threshold,
        ica_ctps_ecg_threshold=config.ica_ctps_ecg_threshold,
        random_state=config.random_state,
        ch_types=config.ch_types,
        l_freq=config.l_freq,
        epochs_decim=config.epochs_decim,
        raw_resample_sfreq=config.raw_resample_sfreq,
        event_repeated=config.event_repeated,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        epochs_metadata_tmin=config.epochs_metadata_tmin,
        epochs_metadata_tmax=config.epochs_metadata_tmax,
        epochs_metadata_keep_first=config.epochs_metadata_keep_first,
        epochs_metadata_keep_last=config.epochs_metadata_keep_last,
        epochs_metadata_query=config.epochs_metadata_query,
        eeg_reference=get_eeg_reference(config),
        eog_channels=config.eog_channels,
        rest_epochs_duration=config.rest_epochs_duration,
        rest_epochs_overlap=config.rest_epochs_overlap,
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run ICA."""
    if config.spatial_filter != 'ica':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            run_ica, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)
