from types import SimpleNamespace
import mne
import os.path
import re
import numpy as np
from mne_bids import BIDSPath

from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
)
from ..._import_data import annotations_to_events, make_epochs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs


def get_input_fnames_sync_eyelink(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> dict:
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )

    et_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="beh",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".asc",
    )


    et_edf_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="beh",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".edf",
    )

    in_files = dict()
    for run in cfg.runs:
        key = f"raw_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)


        key = f"et_run-{run}"
        in_files[key] = et_bids_basename.copy().update(
            run=run
        )
        _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

        key = f"et_edf_run-{run}"
        in_files[key] = et_edf_bids_basename.copy().update(
            run=run
        )
        _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not
    
    return in_files



@failsafe_run(
    get_input_fnames=get_input_fnames_sync_eyelink,
)
def sync_eyelink(
 *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: dict,
) -> dict:
    """Run Sync for Eyelink."""
    import matplotlib.pyplot as plt

    raw_fnames = [in_files.pop(f"raw_run-{run}") for run in cfg.runs]
    et_fnames = [in_files.pop(f"et_run-{run}") for run in cfg.runs]
    et_edf_fnames = [in_files.pop(f"et_edf_run-{run}") for run in cfg.runs]
    
    logger.info(**gen_log_kwargs(message=f"et_fnames {et_fnames}"))
    out_files = dict()
    bids_basename = raw_fnames[0].copy().update(processing=None, split=None, run=None)
    out_files["eyelink"] = bids_basename.copy().update(processing="eyelink", suffix="raw")
    del bids_basename

    
    
    for idx, (run, raw_fname,et_fname,et_edf_fname) in enumerate(zip(cfg.runs, raw_fnames,et_fnames,et_edf_fnames)):
        msg = f"Syncing eyelink data (fake for now) {raw_fname.basename}"
        logger.info(**gen_log_kwargs(message=msg))
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        if not os.path.isfile(et_fname):
            logger.info(**gen_log_kwargs(message=f"Couldn't find {et_fname} file, trying to call edf2asc."))
            if not os.path.isfile(et_edf_fname):
                logger.error(**gen_log_kwargs(message=f"Also didn't find {et_edf_fname} file, one of both need to exist for ET sync."))
            import subprocess
            subprocess.run(["edf2asc", et_edf_fname]) # TODO: Still needs to be tested

        raw_et = mne.io.read_raw_eyelink(et_fname,find_overlaps=True)

        # If the user did not specify a regular expression for the eye-tracking sync events, it is assumed that it's
        # identical to the regex for the EEG sync events
        if not cfg.sync_eventtype_regex_et:
            cfg.sync_eventtype_regex_et = cfg.sync_eventtype_regex
        
        et_sync_times = [annotation["onset"] for annotation in raw_et.annotations if re.search(cfg.sync_eventtype_regex_et,annotation["description"])]
        sync_times    = [annotation["onset"] for annotation in raw.annotations    if re.search(cfg.sync_eventtype_regex,   annotation["description"])]

        assert len(et_sync_times) == len(sync_times),f"Detected eyetracking and EEG sync events were not of equal size ({len(et_sync_times)} vs {len(sync_times)}). Adjust your regular expressions via 'sync_eventtype_regex_et' and 'sync_eventtype_regex' accordingly"
        #logger.info(**gen_log_kwargs(message=f"{et_sync_times}"))
        #logger.info(**gen_log_kwargs(message=f"{sync_times}"))


        # Check whether the eye-tracking data contains nan values. If yes replace them with zeros.
        if np.isnan(raw_et.get_data()).any():

            # Set all nan values in the eye-tracking data to 0 (to make resampling possible)
            # TODO: Decide whether this is a good approach or whether interpolation (e.g. of blinks) is useful
            # TODO: Decide about setting the values (e.g. for blinks) back to nan after synchronising the signals
            np.nan_to_num(raw_et._data, copy=False, nan=0.0)
            logger.info(**gen_log_kwargs(message=f"The eye-tracking data contained nan values. They were replaced with zeros."))

        #mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.05), interpolate_gaze=True)        

        
        # Align the data
        mne.preprocessing.realign_raw(raw, raw_et, sync_times, et_sync_times)


        # Add ET data to EEG
        raw.add_channels([raw_et], force_update_info=True)
        raw._raw_extras.append(raw_et._raw_extras)

        # Also add ET annotations to EEG
        raw.set_annotations(mne.annotations._combine_annotations(raw.annotations,raw_et.annotations,0,raw.first_samp,raw_et.first_samp,raw.info["sfreq"]))


        msg = f"Saving synced data to disk."
        logger.info(**gen_log_kwargs(message=msg))
        raw.save(
            out_files["eyelink"],
            overwrite=True,
            split_naming="bids", # TODO: Find out if we need to add this or not
            split_size=cfg._raw_split_size, # ???
        )
        # no idea what the split stuff is...
        _update_for_splits(out_files, "eyelink") # TODO: Find out if we need to add this or not

    

    # Add to report
    tags = ("sync", "eyelink")
    title = "Synchronize Eyelink"
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        task=cfg.task,
    ) as report:
      
      
        caption = (
           f"The `realign_raw` function from MNE was used to align an Eyelink `asc` file to the M/EEG file."
           f"The Eyelink-data was added as annotations and appended as new channels."
        )
        fig = raw_et.plot(scalings=dict(eyegaze=1e3))
        report.add_figure(
            fig=fig,
            title="Eyelink data",
            section=title,
            caption=caption,
            tags=tags[1],
            replace=True,
        )
        plt.close(fig)
        del caption
    return _prep_out_files(exec_params=exec_params, out_files=out_files)






def get_config(
   *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    #logger.info(**gen_log_kwargs(message=f"config {config}"))

    cfg = SimpleNamespace(
        runs=get_runs(config=config, subject=subject),
        remove_blink_saccades   = config.remove_blink_saccades,
        sync_eventtype_regex    = config.sync_eventtype_regex,
        sync_eventtype_regex_et = config.sync_eventtype_regex_et,
        processing= "filt" if config.regress_artifact is None else "regress",
        _raw_split_size=config._raw_split_size,

        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Sync Eyelink."""
    if not config.sync_eyelink:
        msg = "Skipping, sync_eyelink is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return


    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(sync_eyelink, exec_params=config.exec_params)
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject in get_subjects(config)
            for session in get_sessions(config)
        )
    save_logs(config=config, logs=logs)



