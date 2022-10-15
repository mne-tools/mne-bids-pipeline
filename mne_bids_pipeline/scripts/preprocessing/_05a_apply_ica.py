"""Apply ICA and obtain the cleaned epochs.

Blinks and ECG artifacts are automatically detected and the corresponding ICA
components are removed from the data.
This relies on the ICAs computed in 04-run_ica.py

!! If you manually add components to remove (config.rejcomps_man),
make sure you did not re-run the ICA in the meantime. Otherwise (especially if
the random state was not set, or you used a different machine, the component
order might differ).

"""

import itertools
from types import SimpleNamespace

import pandas as pd
import mne
from mne.preprocessing import read_ica
from mne.report import Report

from mne_bids import BIDSPath

from ..._config_utils import (
    get_subjects, get_sessions, get_task, get_datatype, get_deriv_root,
)
from ..._logging import gen_log_kwargs, logger
from ..._run import failsafe_run, _update_for_splits, _script_path, save_logs
from ..._parallel import parallel_func, get_parallel_backend
from ..._reject import _get_reject


def get_input_fnames_apply_ica(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
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
    in_files['ica'] = bids_basename.copy().update(
        suffix='ica', extension='.fif')
    in_files['components'] = bids_basename.copy().update(
        processing='ica', suffix='components', extension='.tsv')
    in_files['epochs'] = bids_basename.copy().update(
        suffix='epo', extension='.fif')
    return in_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_apply_ica)
def apply_ica(*, cfg, subject, session, in_files):
    bids_basename = in_files['ica'].copy().update(processing=None)
    out_files = dict()
    out_files['epochs'] = in_files['epochs'].copy().update(processing='ica')
    out_files['report'] = bids_basename.copy().update(
        processing='ica', suffix='report', extension='.html')

    title = f'ICA artifact removal – sub-{subject}'
    if session is not None:
        title += f', ses-{session}'
    if cfg.task is not None:
        title += f', task-{cfg.task}'

    # Load ICA.
    msg = f"Reading ICA: {in_files['ica']}"
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))
    ica = read_ica(fname=in_files.pop('ica'))

    # Select ICs to remove.
    tsv_data = pd.read_csv(in_files.pop('components'), sep='\t')
    ica.exclude = (tsv_data
                   .loc[tsv_data['status'] == 'bad', 'component']
                   .to_list())

    # Load epochs to reject ICA components.
    msg = (f'Input: {in_files["epochs"].basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = (f'Output: {out_files["epochs"].basename}')
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(in_files.pop('epochs'), preload=True)
    ica_reject = _get_reject(
        subject=subject,
        session=session,
        reject=cfg.ica_reject,
        ch_types=cfg.ch_types,
        param='ica_reject',
    )
    epochs.drop_bad(ica_reject)

    # Now actually reject the components.
    msg = f'Rejecting ICs: {", ".join([str(ic) for ic in ica.exclude])}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_cleaned = ica.apply(epochs.copy())  # Copy b/c works in-place!

    msg = 'Saving reconstructed epochs after ICA.'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    epochs_cleaned.save(
        out_files['epochs'], overwrite=True, split_naming='bids',
        split_size=cfg._epochs_split_size)
    _update_for_splits(out_files, 'epochs')

    # Compare ERP/ERF before and after ICA artifact rejection. The evoked
    # response is calculated across ALL epochs, just like ICA was run on
    # all epochs, regardless of their respective experimental condition.
    #
    # We apply baseline correction here to (hopefully!) make the effects of
    # ICA easier to see. Otherwise, individual channels might just have
    # arbitrary DC shifts, and we wouldn't be able to easily decipher what's
    # going on!
    report = Report(
        out_files['report'], title=title, verbose=False)
    picks = ica.exclude if ica.exclude else None
    report.add_ica(
        ica=ica,
        title='Effects of ICA cleaning',
        inst=epochs.copy().apply_baseline(cfg.baseline),
        picks=picks
    )
    report.save(
        out_files['report'], overwrite=True, open_browser=cfg.interactive)

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
        interactive=config.interactive,
        baseline=config.baseline,
        ica_reject=config.ica_reject,
        ch_types=config.ch_types,
        _epochs_split_size=config._epochs_split_size,
    )
    return cfg


def main():
    """Apply ICA."""
    import config
    if not config.spatial_filter == 'ica':
        msg = 'Skipping …'
        with _script_path(__file__):
            logger.info(**gen_log_kwargs(message=msg, emoji='skip'))
        return

    with get_parallel_backend(config):
        parallel, run_func = parallel_func(apply_ica, config=config)
        logs = parallel(
            run_func(
                cfg=get_config(config=config),
                subject=subject,
                session=session)
            for subject, session in
            itertools.product(
                get_subjects(config),
                get_sessions(config)
            )
        )
    save_logs(config=config, logs=logs)


if __name__ == '__main__':
    main()
