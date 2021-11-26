"""
===========
04. Run ICA
===========
This fits ICA on epoched data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.

Before performing ICA, we reject epochs based on peak-to-peak amplitude above
the 'ica_reject' to filter massive non-biological artifacts.

To actually remove designated ICA components from your data, you will have to
run 05a-apply_ica.py.
"""

import sys
import itertools
import logging
from typing import List, Optional, Iterable, Tuple
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import pandas as pd
import numpy as np

import mne
from mne.utils import BunchConst
from mne.report import Report
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import (make_epochs, gen_log_kwargs, on_error, failsafe_run,
                    annotations_to_events)

logger = logging.getLogger('mne-bids-pipeline')


def filter_for_ica(
    *,
    cfg,
    raw: mne.io.BaseRaw,
    subject: str,
    session: str,
    run: Optional[str] = None
) -> None:
    """Apply a high-pass filter if needed."""
    if cfg.ica_l_freq is None:
        msg = (f'Not applying high-pass filter (data is already filtered, '
               f'cutoff: {raw.info["highpass"]} Hz).')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
    else:
        msg = f'Applying high-pass filter with {cfg.ica_l_freq} Hz cutoff …'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        raw.filter(l_freq=cfg.ica_l_freq, h_freq=None)


def fit_ica(
    *,
    cfg,
    epochs: mne.BaseEpochs,
    subject: str,
    session: str,
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
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    return ica


def make_ecg_epochs(
    *,
    cfg,
    raw_path: BIDSPath,
    subject: str,
    session: str,
    run: Optional[str] = None,
    n_runs: int
) -> Optional[mne.BaseEpochs]:
    # ECG either needs an ecg channel, or avg of the mags (i.e. MEG data)
    raw = mne.io.read_raw(raw_path, preload=False)

    if ('ecg' in raw.get_channel_types() or 'meg' in cfg.ch_types or
            'mag' in cfg.ch_types):
        msg = 'Creating ECG epochs …'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))

        # We want to extract a total of 5 min of data for ECG epochs generation
        # (across all runs)
        total_ecg_dur = 5 * 60
        ecg_dur_per_run = total_ecg_dur / n_runs
        t_mid = (raw.times[-1] - raw.times[0]) / 2
        raw = raw.crop(
            tmin=t_mid - 1/2 * ecg_dur_per_run,
            tmax=t_mid + 1/2 * ecg_dur_per_run
        ).load_data()

        ecg_epochs = create_ecg_epochs(raw,
                                       baseline=(None, -0.2),
                                       tmin=-0.5, tmax=0.5)
        del raw  # Free memory

        if len(ecg_epochs) == 0:
            msg = ('No ECG events could be found. Not running ECG artifact '
                   'detection.')
            logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                         session=session, run=run))
            ecg_epochs = None
    else:
        msg = ('No ECG or magnetometer channels are present. Cannot '
               'automate artifact detection for ECG')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        ecg_epochs = None

    return ecg_epochs


def make_eog_epochs(
    *,
    raw: mne.io.BaseRaw,
    eog_channels: Optional[Iterable[str]],
    subject: str,
    session: str,
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
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))

        eog_epochs = create_eog_epochs(raw, ch_name=ch_names,
                                       baseline=(None, -0.2))

        if len(eog_epochs) == 0:
            msg = ('No EOG events could be found. Not running EOG artifact '
                   'detection.')
            logger.warning(**gen_log_kwargs(message=msg,
                                            subject=subject,
                                            session=session, run=run))
            eog_epochs = None
    else:
        msg = ('No EOG channel is present. Cannot automate IC detection '
               'for EOG')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        eog_epochs = None

    return eog_epochs


def detect_bad_components(
    *,
    cfg,
    which: Literal['eog', 'ecg'],
    epochs: mne.BaseEpochs,
    ica: mne.preprocessing.ICA,
    subject: str,
    session: str
) -> Tuple[List[int], np.ndarray]:
    artifact = which.upper()
    msg = f'Performing automated {artifact} artifact detection …'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    if which == 'eog':
        inds, scores = ica.find_bads_eog(
            epochs,
            threshold=cfg.ica_eog_threshold
        )
    else:
        inds, scores = ica.find_bads_ecg(
            epochs, method='ctps',
            threshold=cfg.ica_ctps_ecg_threshold
        )

    if not inds:
        adjust_setting = ('ica_eog_threshold' if which == 'eog'
                          else 'ica_ctps_ecg_threshold')
        warn = (f'No {artifact}-related ICs detected, this is highly '
                f'suspicious. A manual check is suggested. You may wish to '
                f'lower "{adjust_setting}".')
        logger.warning(**gen_log_kwargs(message=warn,
                                        subject=subject,
                                        session=session))
    else:
        msg = (f'Detected {len(inds)} {artifact}-related ICs in '
               f'{len(epochs)} {artifact} epochs.')
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

    return inds, scores


@failsafe_run(on_error=on_error, script_path=__file__)
def run_ica(*, cfg, subject, session=None):
    """Run ICA."""
    bids_basename = BIDSPath(subject=subject,
                             session=session,
                             task=cfg.task,
                             acquisition=cfg.acq,
                             recording=cfg.rec,
                             space=cfg.space,
                             datatype=cfg.datatype,
                             root=cfg.deriv_root,
                             check=False)

    raw_fname = bids_basename.copy().update(processing='filt', suffix='raw')
    ica_fname = bids_basename.copy().update(suffix='ica', extension='.fif')
    ica_components_fname = bids_basename.copy().update(processing='ica',
                                                       suffix='components',
                                                       extension='.tsv')
    report_fname = bids_basename.copy().update(processing='ica+components',
                                               suffix='report',
                                               extension='.html')

    # Generate a list of raw data paths (i.e., paths of individual runs)
    # we want to create epochs from.
    raw_fnames = []
    for run in cfg.runs:
        raw_fname.update(run=run)
        if raw_fname.copy().update(split='01').fpath.exists():
            raw_fname.update(split='01')

        raw_fnames.append(raw_fname.copy())

    # Generate a unique event name -> event code mapping that can be used
    # across all runs.
    event_name_to_code_map = annotations_to_events(raw_paths=raw_fnames)

    # Now, generate epochs from each individual run
    eog_epochs_all_runs = None
    ecg_epochs_all_runs = None

    for idx, (run, raw_fname) in enumerate(
        zip(cfg.runs, raw_fnames)
    ):
        msg = f'Loading filtered raw data from {raw_fname} and creating epochs'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))

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
        if cfg.resample_sfreq is not None:
            assert np.allclose(raw.info['sfreq'], cfg.resample_sfreq)
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
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session, run=run))
        epochs = make_epochs(
            raw=raw,
            event_id=event_id,
            tmin=cfg.epochs_tmin,
            tmax=cfg.epochs_tmax,
            event_repeated=cfg.event_repeated,
            decim=cfg.decim
        )

        # Only keep epochs that will be analyzed -> Keeps ICA in sync with
        # epochs generated in the make_epochs script (save preserves memory)!
        if cfg.task != 'rest':
            if isinstance(cfg.conditions, dict):
                conditions = list(cfg.conditions.keys())
            else:
                conditions = cfg.conditions
            epochs = epochs[conditions]

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
    msg = f'Using PTP rejection thresholds: {cfg.ica_reject}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    # Reject epochs based on peak-to-peak amplitude
    epochs.drop_bad(reject=cfg.ica_reject)
    if epochs_eog is not None:
        epochs_eog.drop_bad(reject=cfg.ica_reject)
    if epochs_ecg is not None:
        epochs_ecg.drop_bad(reject=cfg.ica_reject)

    # Now actually perform ICA.
    msg = 'Calculating ICA solution.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
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
            cfg=cfg, which='ecg', epochs=epochs_ecg,
            ica=ica,
            subject=subject,
            session=session
        )
    else:
        ecg_ics = ecg_scores = []

    if epochs_eog:
        eog_ics, eog_scores = detect_bad_components(
            cfg=cfg, which='eog', epochs=epochs_eog,
            ica=ica,
            subject=subject,
            session=session
        )
    else:
        eog_ics = eog_scores = []

    # Save ICA to disk.
    # We also store the automatically identified ECG- and EOG-related ICs.
    msg = 'Saving ICA solution and detected artifacts to disk.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    ica.exclude = sorted(set(ecg_ics + eog_ics))
    ica.save(ica_fname, overwrite=True)

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

    tsv_data.to_csv(ica_components_fname, sep='\t', index=False)

    # Lastly, add info about the epochs used for the ICA fit, and plot all ICs
    # for manual inspection.
    msg = 'Adding diagnostic plots for all ICA components to the HTML report …'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    report = Report(info_fname=epochs, title=title, verbose=False)
    report.add_epochs(
        epochs=epochs,
        title='Epochs used for ICA fitting'
    )

    ecg_evoked = None if epochs_ecg is None else epochs_ecg.average()
    eog_evoked = None if epochs_eog is None else epochs_eog.average()
    ecg_scores = None if len(ecg_scores) == 0 else ecg_scores
    eog_scores = None if len(eog_scores) == 0 else eog_scores

    report.add_ica(
        ica=ica,
        title='ICA cleaning',
        inst=epochs,
        ecg_evoked=ecg_evoked,
        eog_evoked=eog_evoked,
        ecg_scores=ecg_scores,
        eog_scores=eog_scores
    )

    msg = (f"ICA completed. Please carefully review the extracted ICs in the "
           f"report {report_fname.basename}, and mark all components you wish "
           f"to reject as 'bad' in {ica_components_fname.basename}")
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    report.save(report_fname, overwrite=True, open_browser=cfg.interactive)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        conditions=config.conditions,
        task=config.get_task(),
        datatype=config.get_datatype(),
        runs=config.get_runs(subject=subject),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        interactive=config.interactive,
        ica_l_freq=config.ica_l_freq,
        ica_algorithm=config.ica_algorithm,
        ica_n_components=config.ica_n_components,
        ica_max_iterations=config.ica_max_iterations,
        ica_decim=config.ica_decim,
        ica_reject=config.get_ica_reject(),
        ica_eog_threshold=config.ica_eog_threshold,
        ica_ctps_ecg_threshold=config.ica_ctps_ecg_threshold,
        random_state=config.random_state,
        ch_types=config.ch_types,
        l_freq=config.l_freq,
        decim=config.decim,
        resample_sfreq=config.resample_sfreq,
        event_repeated=config.event_repeated,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        eeg_reference=config.get_eeg_reference(),
        eog_channels=config.eog_channels
    )
    return cfg


def main():
    """Run ICA."""
    if not config.spatial_filter == 'ica':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func, _ = parallel_func(run_ica,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(subject=subject), subject=subject,
                 session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
