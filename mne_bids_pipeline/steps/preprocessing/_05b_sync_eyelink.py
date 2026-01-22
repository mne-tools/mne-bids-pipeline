from types import SimpleNamespace
import mne
import os.path
import re
import numpy as np
from mne_bids import BIDSPath
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
    _get_ssrt,
)
from ..._import_data import annotations_to_events, make_epochs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import get_parallel_backend, parallel_func
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs

def _check_HEOG_ET_vars(cfg):
    # helper function for sorting out heog and et channels
    bipolar = False
    if isinstance(cfg.sync_heog_ch, tuple):
        heog_ch = "bi_HEOG"
        bipolar = True
    else:
        heog_ch = cfg.sync_heog_ch
    
    if isinstance(cfg.sync_et_ch, tuple):
        et_ch = list(cfg.sync_et_ch)
    else:
        et_ch = [cfg.sync_et_ch]
    
    return heog_ch, et_ch, bipolar

def _mark_calibration_as_bad(raw, cfg):
    # marks recalibration beginnings and ends as one bad segment
    cur_idx = None
    cur_start_time = 0.
    last_status = None
    for annot in raw.annotations:
        calib_match = re.match(cfg.sync_calibration_string, annot["description"])
        if not calib_match: continue
        calib_status, calib_idx = calib_match.group(1), calib_match.group(2)
        if calib_idx  == cur_idx and calib_status == "end":
            duration = annot["onset"] - cur_start_time
            raw.annotations.append(cur_start_time, duration, f"BAD_Recalibrate {calib_idx}")
            cur_idx, cur_start_time = None, 0.
        elif calib_status == "start" and cur_idx is None:
            cur_idx = calib_idx
            cur_start_time = annot["onset"]
        elif calib_status == last_status:
            logger.info(**gen_log_kwargs(message=f"Encountered apparent duplicate calibration event ({calib_status}, {calib_idx}) - skipping"))
        elif calib_status == "start" and cur_idx is not None:
            raise ValueError(f"Annotation {annot["description"]} could not be assigned membership"
                             f"")
        last_status = calib_status
        
    return raw
        

def get_input_fnames_sync_eyelink(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> dict:
    
    # Get from config file whether `task` is specified in the et file name
    if cfg.et_has_task == True:
        et_task = cfg.task
    else:
        et_task = None

    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        run=run,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )

    et_asc_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="misc",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".asc",
    )

    et_edf_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="misc",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".edf",
    )

    in_files = dict()

    key = f"raw_run-{run}"
    in_files[key] = bids_basename.copy().update(
        processing=cfg.processing, suffix="raw"
    )

    et_bids_basename_temp = et_asc_bids_basename.copy()

    if cfg.et_has_run:
        et_bids_basename_temp.update(run=run)

    # _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

    if not os.path.isfile(et_bids_basename_temp):
        logger.info(**gen_log_kwargs(message=f"Couldn't find {et_bids_basename_temp} file. If edf file exists, edf2asc will be called."))

        et_bids_basename_temp = et_edf_bids_basename.copy()

        if cfg.et_has_run:
            et_bids_basename_temp.update(run=run)

        # _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

        if not os.path.isfile(et_bids_basename_temp):
            logger.error(**gen_log_kwargs(message=f"Also didn't find {et_bids_basename_temp} file, one of both needs to exist for ET sync."))
            raise FileNotFoundError(f"For run {run}, could neither find .asc or .edf eye-tracking file. Please double-check the file names.")

    key = f"et_run-{run}"
    in_files[key] = et_bids_basename_temp
  
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
    run: str,
    task: str | None,
    in_files: dict,
) -> dict:
    
    """Run Sync for Eyelink."""
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    raw_fname = in_files.pop(f"raw_run-{run}")
    et_fname = in_files.pop(f"et_run-{run}")
    logger.info(**gen_log_kwargs(message=f"Found the following eye-tracking files: {et_fname}"))
    out_files = dict()
    bids_basename = raw_fname.copy().update(processing=None, split=None) #TODO: Do we need the `split=None` here?
    out_files["eyelink_eeg"] = bids_basename.copy().update(processing="eyelink", suffix="raw")
    del bids_basename

    # Ideally, this would be done in one of the previous steps where all folders are created (in `_01_init_derivatives_dir.py`). 
    logger.info(**gen_log_kwargs(message=f"Create `misc` folder for eye-tracking events."))
    out_dir_misc = cfg.deriv_root / f"sub-{subject}"
    if session is not None:
        out_dir_misc /= f"ses-{session}"

    out_dir_misc /= "misc"
    out_dir_misc.mkdir(exist_ok=True, parents=True) # TODO: Check whether the parameter settings make sense or if there is a danger that something could be accidentally overwritten

    out_files["eyelink_et_events"] = et_fname.copy().update(root=cfg.deriv_root, suffix="et_events", extension=".tsv")
    
    msg = f"Syncing Eyelink ({et_fname.basename}) and EEG data ({raw_fname.basename})."
    logger.info(**gen_log_kwargs(message=msg))
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    et_format = et_fname.extension

    if not et_format == '.asc':
        assert et_format == '.edf', "ET file is neither an `.asc` nor an `.edf`. This should not have happened."
        logger.info(**gen_log_kwargs(message=f"Converting {et_fname} file to `.asc` using edf2asc."))
        import subprocess
        subprocess.run(["edf2asc", et_fname]) # TODO: Still needs to be tested
        et_fname.update(extension='.asc')

    raw_et = mne.io.read_raw_eyelink(et_fname, find_overlaps=False) # TODO: Make find_overlaps optional

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
        # TODO: Tip: With `mne.preprocessing.annotate_nan` you could get the timings comparatively easy, and then after `realign_raw` put nans on top.
        np.nan_to_num(raw_et._data, copy=False, nan=0.0)
        logger.info(**gen_log_kwargs(message=f"The eye-tracking data contained nan values. They were replaced with zeros."))

    #mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.05), interpolate_gaze=True)        

    # Align the data
    mne.preprocessing.realign_raw(raw, raw_et, sync_times, et_sync_times)

    # Add ET data to EEG
    raw.add_channels([raw_et], force_update_info=True)

    # Also add ET annotations to EEG
    # first mark et sync event descriptions so we can differentiate them later
    # TODO: For now all ET events will be marked with ET and added to the EEG annotations, maybe later filter for certain events only
    raw_et.annotations.description = np.array(list(map(lambda desc: "ET_" + desc, raw_et.annotations.description)))
    raw.set_annotations(mne.annotations._combine_annotations(raw.annotations,
                                                                raw_et.annotations,
                                                                0,
                                                                raw.first_samp,
                                                                raw_et.first_samp,
                                                                raw.info["sfreq"]))

    msg = f"Saving synced data to disk."
    logger.info(**gen_log_kwargs(message=msg))
    raw.save(
        out_files["eyelink_eeg"],
        overwrite=True,
        split_naming="bids", # TODO: Find out if we need to add this or not
        split_size=cfg._raw_split_size, # ???
    )
    # no idea what the split stuff is...
    # _update_for_splits(out_files, "eyelink_eeg") # TODO: Find out if we need to add this or not

    # Extract and concatenate eye-tracking event data frames
    et_dfs = raw_et._raw_extras[0]["dfs"]
    df_list = [] # List to collect extracted data frames before concatenation

    # Extract fixations, saccades and blinks data frames
    for df_name, trial_type in zip(["fixations", "saccades", "blinks"], ["fixation", "saccade", "blink"]):
        df = et_dfs[df_name]
        df["trial_type"] = trial_type
        df_list.append(df)

    et_combined_df = pd.concat(df_list, ignore_index=True)
    et_combined_df.rename(columns={"time":"onset"}, inplace=True)
    et_combined_df.sort_values(by="onset", inplace=True, ignore_index=True)
    et_combined_df = et_combined_df[ # Adapt column order
        [
            "onset", # needs to be first (BIDS convention)
            "duration",
            "end_time",
            "trial_type",
            "eye",
            "fix_avg_x",
            "fix_avg_y",
            "fix_avg_pupil_size",
            "sacc_start_x",
            "sacc_start_y",
            "sacc_end_x",
            "sacc_end_y",
            "sacc_visual_angle",
            "peak_velocity"
        ]
    ] 

    # Synchronize eye-tracking events with EEG data

    # Recalculate regression coefficients (because the realign_raw function does not output them)
    # Code snippet from `mne.preprocessing.realign_raw` function:
    # https://github.com/mne-tools/mne-python/blob/b44c46ae7f9b6ffc5318b5d64f12906c1f2d875c/mne/preprocessing/realign.py#L69-L71
    poly = Polynomial.fit(x=et_sync_times, y=sync_times, deg=1)
    converted = poly.convert(domain=(-1, 1))
    [zero_ord, first_ord] = converted.coef
    # print(zero_ord, first_ord)

    # Synchronize time stamps of ET events
    et_combined_df["onset"] = (et_combined_df["onset"] * first_ord + zero_ord)
    et_combined_df["end_time"] = (et_combined_df["end_time"] * first_ord + zero_ord)
    # TODO: To be super correct, we would need to recalculate duration column as well - but typically the slope is so close to "1" that this would typically result in <1ms differences

    msg = f"Saving synced eye-tracking events to disk."
    logger.info(**gen_log_kwargs(message=msg))
    et_combined_df.to_csv(out_files["eyelink_et_events"], sep="\t", index=False)

    # Add to report
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 19.2))
    msg = f"Adding figure to report."
    logger.info(**gen_log_kwargs(message=msg))
    tags = ("sync", "eyelink")
    title = "Synchronize Eyelink"
    caption = (
           f"The `realign_raw` function from MNE was used to align an Eyelink `asc` file to the M/EEG file."
           f"The Eyelink-data was added as annotations and appended as new channels."
        )
    if cfg.sync_heog_ch is None or cfg.sync_et_ch is None:
        # we need both an HEOG channel and ET channel specified to do cross-correlation
        msg = f"HEOG and/or ET channel not specified; cannot produce cross-correlation for report."
        logger.info(**gen_log_kwargs(message=msg))
        caption += "\nHEOG and/or eye tracking channels were not specified and no cross-correlation was performed."
        axes[0,0].text(0.5, 0.5, 'HEOG/ET cross-correlation unavailable', fontsize=34,
                       horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes)
        axes[0,0].axis("off")
    else:
        # return _prep_out_files(exec_params=exec_params, out_files=out_files)
        # calculate cross correlation of HEOG with ET
        heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
        if bipolar:
            # create bipolar HEOG
            raw = mne.set_bipolar_reference(raw, *cfg.sync_heog_ch, ch_name=heog_ch, drop_refs=False)
        raw.filter(l_freq=cfg.sync_heog_highpass, h_freq=cfg.sync_heog_lowpass, picks=heog_ch) # get rid of drift and high freq noise
        _mark_calibration_as_bad(raw, cfg)
        # extract HEOG and ET as arrays
        heog_array = raw.get_data(picks=[heog_ch], reject_by_annotation="omit")
        et_array = raw.get_data(picks=et_ch, reject_by_annotation="omit")
        if len(et_array) > 1:
            et_array = et_array.mean(axis=0, keepdims=True)
        # cross correlate them
        corr = correlate(heog_array[0], et_array[0], mode="same") / heog_array.shape[1]
        # plot cross correlation
        # figure out how much we plot
        midpoint = len(corr) // 2
        plot_samps = (-cfg.sync_plot_samps, cfg.sync_plot_samps) if isinstance(cfg.sync_plot_samps, int) else cfg.sync_plot_samps
        if isinstance(plot_samps, tuple):
            x_range = np.arange(plot_samps[0], plot_samps[1])
            y_range = np.arange(midpoint+plot_samps[0], midpoint+plot_samps[1])
        else: # None
            y_range = np.arange(len(corr))
            x_range = y_range - midpoint
        # plot
        axes[0,0].plot(x_range, corr[y_range], color="black")
        axes[0,0].axvline(linestyle="--", alpha=0.3)
        axes[0,0].set_title("Cross correlation HEOG and ET")
        axes[0,0].set_xlabel("Samples")
        axes[0,0].set_ylabel("X correlation")
        # calculate delay
        delay_idx = abs(corr).argmax() - midpoint
        delay_time = delay_idx * (raw.times[1] - raw.times[0])
        caption += f"\nThere was an estimated synchronisation delay of {delay_idx} samples ({delay_time:.3f} seconds.)"
    
    # regression between synced events
    # we assume here that these annotations are sequential pairs of the same event in raw and et. otherwise this will break
    raw_onsets = [annot["onset"] for annot in raw.annotations if re.match("^(?!.*ET_)"+cfg.sync_eventtype_regex, annot["description"])]
    et_onsets = [annot["onset"] for annot in raw.annotations if re.match("ET_"+cfg.sync_eventtype_regex_et, annot["description"])]
 
    if len(raw_onsets) != len(et_onsets):
        raise ValueError(f"Lengths of raw {len(raw_onsets)} and ET {len(et_onsets)} onsets do not match.")
    # regress and plot
    coef = np.polyfit(raw_onsets, et_onsets, 1)
    preds = np.poly1d(coef)(raw_onsets)
    resids = et_onsets - preds
    axes[0,1].plot(raw_onsets, et_onsets, "o", alpha=0.3, color="black")
    axes[0,1].plot(raw_onsets, preds, "--k")
    axes[0,1].set_title("Regression")
    axes[0,1].set_xlabel("Raw onsets (seconds)")
    axes[0,1].set_ylabel("ET onsets (seconds)")
    # residuals
    axes[1,0].plot(np.arange(len(resids)), resids, "o", alpha=0.3, color="black")
    axes[1,0].axhline(linestyle="--")
    axes[1,0].set_title("Residuals")
    axes[1,0].set_ylabel("Residual (seconds)")
    axes[1,0].set_xlabel("Samples")
    # histogram of distances between events in time
    axes[1,1].hist(np.array(raw_onsets) - np.array(et_onsets), bins=11, range=(-5,5), color="black")
    axes[1,1].set_title("Raw - ET event onset distances histogram")
    axes[1,1].set_xlabel("milliseconds")
    # this doesn't seem to help, though it should...
    fig.tight_layout()

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=cfg.task,
    ) as report:
        caption = caption
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
        et_has_run = config.et_has_run,
        et_has_task = config.et_has_task,
        sync_eventtype_regex    = config.sync_eventtype_regex,
        sync_eventtype_regex_et = config.sync_eventtype_regex_et,
        sync_heog_ch = config.sync_heog_ch,
        sync_et_ch = config.sync_et_ch,
        sync_heog_highpass = config.sync_heog_highpass,
        sync_heog_lowpass = config.sync_heog_lowpass,
        sync_plot_samps = config.sync_plot_samps,
        sync_calibration_string = config.sync_calibration_string,
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

    ssrt = _get_ssrt(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(sync_eyelink, exec_params=config.exec_params, n_iter=len(ssrt))
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )
    save_logs(config=config, logs=logs)