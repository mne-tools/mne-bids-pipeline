## v1.10.0 (unreleased)

### :new: New features & enhancements

- Support for using `mne-icalabel` to automatically label ICA components. This requires the `mne-icalabel` package to be installed. (#1018 and #812 by @jschepers, @behinger, @SNaber, @hoechenberger, and @larsoner)
- It is now possible to use separate MRIs for each session within a subject, as in longitudinal studies. This is achieved by creating separate "subject" folders for each subject-session combination, with the naming convention `sub-XXX_ses-YYY`, in the freesurfer `SUBJECTS_DIR`. (#987 by @drammock)
- New config option [`allow_missing_sessions`][mne_bids_pipeline._config.allow_missing_sessions] allows to continue when not all sessions are present for all subjects. (#1000 by @drammock)
- New config option [`mf_extra_kws`][mne_bids_pipeline._config.mf_extra_kws] passes additional keyword arguments to `mne.preprocessing.maxwell_filter`. (#1038 by @drammock)
- New value `"twa"` for config option [`mf_destination`][mne_bids_pipeline._config.mf_destination], to use the time-weighted average head position across runs as the destination position. (#1043 and #1055 by @drammock)
- New config options [`mf_cal_missing`][mne_bids_pipeline._config.mf_cal_missing] and [`mf_ctc_missing`][mne_bids_pipeline._config.mf_ctc_missing] for handling missing calibration and cross-talk files (#1057 by @harrisonritz)
- New config options [`find_bad_channels_extra_kws`][mne_bids_pipeline._config.find_bad_channels_extra_kws], [`notch_extra_kws`][mne_bids_pipeline._config.notch_extra_kws], and [`bandpass_extra_kws`][mne_bids_pipeline._config.bandpass_extra_kws] to pass additional keyword arguments to `mne.preprocessing.find_bad_channels_maxwell`, `mne.filter.notch_filter`, and `mne.filter.filter_data` respectively (#1061 by @harrisonritz)
- Bad channel reports now say which channels were marked as bad in the original data and which were detected automatically by the pipeline. (#1138 by @larsoner)
- Config option [`ssp_ecg_channel`][mne_bids_pipeline._config.ssp_ecg_channel] now allows dict values, for setting a different channel name for each subject/session (#1062 by @drammock)
- New config option [`epochs_custom_metadata`][mne_bids_pipeline._config.epochs_custom_metadata] allows for custom metadata when creating epochs. (#1088 by @harrisonritz)
- New config option [`process_raw_clean`] to allow not fully cleaning raw data to save computing resources (#1136 by @larsoner)
- Automatically avoid using `joblib.Parallel` when only one iteration is needed (and `dask` is not being used), allowing multithreaded linear algebra operations to be used (#1136 by @larsoner)
- Add support for subject and session-specific EOG channels in config and change ssp_ecg_channel to accept 'default' key when dict (#1151 by @skjerns)
- Add support for computing data rank using `mne.compute_rank` via [`cov_rank`][mne_bids_pipeline._config.cov_rank] (#1159 by @larsoner)

### :warning: Behavior changes

- The pipeline will now raise an error if a loaded `SourceSpaces` object has a `._subject` attribute different from what the pipeline expects / would have used if creating the `SourceSpaces` anew. (#1056 by @drammock)
- Maxwell filtering steps now use the union of bad channels across all tasks and runs to help ensure more consistent rank and behavior (#1158 by @larsoner)
- The path passed to `noise_cov` callables is no longer the evoked dataset but instead epochs. Note that the epochs have `processing="clean"` in the `BIDSPath`. (#1159 by @larsoner)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :package: Requirements)

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- Empty room matching is now done for all sessions (previously only for the first session) for each subject. (#976 by @drammock)
- [`noise_cov_method`][mne_bids_pipeline._config.noise_cov_method] is now properly used for noise covariance estimation from raw data (#1010 by @larsoner)
- When running the pipeline with [`mf_filter_chpi`][mne_bids_pipeline._config.mf_filter_chpi] enabled (#977 by @drammock and @larsoner):

    1. Emptyroom files that lack cHPI channels will now be processed (for line noise only) instead of raising an error.
    2. cHPI filtering is now performed before movement compensation.

- Fix bug where the `config.proc` parameter was not used properly during forward model creation (#1014 by @larsoner)
- Fix bug where emptyroom recordings containing EEG channels would crash the pipeline during maxwell filtering (#1040 by @drammock)
- Fix bug where only having mag sensors would crash compute_rank during maxwell filtering or epoching (#1061 and #1069 by @harrisonritz)
- Improvements to template config file generation (#1074 by @drammock)
- Fix bug where `mf_int_order` wasn't passed to `maxwell_filter`. Added config option for `mf_ext_order`. (#1092 by @harrisonritz)
- Sanitize report tags that contain `"` or `'`, e.g., for certain metadata contrasts (#1097 by @harrisonritz)
- `sub-average` is no longer run when ``--subject`` is passed to the pipeline command line interface, allowing easier and more correct single-subject iteration (#1133 by @larsoner)
- Fix bug where time-weighted head position was not cached properly (#1135 by @larsoner)
- Fix bug where some config options were unnecessarily passed to steps, possibly leading to unnecessary cache misses (#1152 by @larsoner)
- Fix bug where the [`config.data_type`][mne_bids_pipeline._config.data_type] channels were not properly picked during ICA fitting, leading to potential errors during ICA application (#1156 by @larsoner)
- Fix bug where a nonexistent evoked `BIDSPath` was passed to user-defined [`noise_cov`][mne_bids_pipeline._config.noise_cov] callables, a path to epochs is now passed instead (#1159 by @larsoner)
- Fix bug where ICA pre-filtering was only applied to run-1, now filters all runs (#1164 by @Okimoka)

### :books: Documentation

- Choose the theme (dark of light) automatically based on the user's operating system setting (#979 by @hoechenberger)
- Bundle all previously-external JavaScript to better preserve users' privacy (#982 by @hoechenberger)
- Document the need for offscreen rendering support when running on headless servers (#997 by @drammock)
- Add more info on test suite installation and running (#1154 by @skjerns)

### :medical_symbol: Code health and infrastructure

- Switch from using relative to using absolute imports (#969 by @hoechenberger)
- Enable strict type checking via mypy (#995, #1013, #1016 by @larsoner)
- Improve logging messages in maxwell filtering steps. (#893 by @drammock)
- Validate extra config params passed in during testing. (#1044 by @drammock)
- New testing/example dataset "funloc" added. (#1045 by @drammock)
- Bugfixes and better testing of session-specific MRIs. (#1039 and #1067 by @drammock)
- Drop legacy function `inst.pick_types` in favor of `inst.pick` (#1073 by @PierreGtch)
- Moved to `ty`, `lefthook`, and `dependency-groups` for development (#1166 by @larsoner)
