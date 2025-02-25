## v1.10.0 (unreleased)

### :new: New features & enhancements

- It is now possible to use separate MRIs for each session within a subject, as in longitudinal studies. This is achieved by creating separate "subject" folders for each subject-session combination, with the naming convention `sub-XXX_ses-YYY`, in the freesurfer `SUBJECTS_DIR`. (#987 by @drammock)
- New config option [`allow_missing_sessions`][mne_bids_pipeline._config.allow_missing_sessions] allows to continue when not all sessions are present for all subjects. (#1000 by @drammock)
- New config option [`mf_extra_kws`][mne_bids_pipeline._config.mf_extra_kws] passes additional keyword arguments to `mne.preprocessing.maxwell_filter`. (#1038 by @drammock)
- New value `"twa"` for config option [`mf_destination`][mne_bids_pipeline._config.mf_destination], to use the time-weighted average head position across runs as the destination position. (#1043 and #1055 by @drammock)
- New config option [`mf_cal_missing`][mne_bids_pipeline._config.mf_cal_missing] and [`mf_ctc_missing`][mne_bids_pipeline._config.mf_ctc_missing] for handling missing calibration and cross-talk files ['raise', 'warn', 'ignore'] (#1057 by @harrisonritz)

### :warning: Behavior changes

- The pipeline will now raise an error if a loaded `SourceSpaces` object has a `._subject` attribute different from what the pipeline expects / would have used if creating the `SourceSpaces` anew. (#1056 by @drammock)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :package: Requirements)

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- Empty room matching is now done for all sessions (previously only for the first session) for each subject. (#976 by @drammock)
- [`noise_cov_method`][mne_bids_pipeline._config.noise_cov_method] is now properly used for noise covariance estimation from raw data (#1010 by @larsoner)
- When running the pipeline with [`mf_filter_chpi`][mne_bids_pipeline._config.mf_filter_chpi] enabled (#977 by @drammock and @larsoner):

    1. Emptyroom files that lack cHPI channels will now be processed (for line noise only) instead of raising an error.
    2. cHPI filtering is now performed before movement compensation.

- Fix bug where the ``config.proc`` parameter was not used properly during forward model creation (#1014 by @larsoner)
- Fix bug where emptyroom recordings containing EEG channels would crash the pipeline during maxwell filtering (#1040 by @drammock)


### :books: Documentation

- Choose the theme (dark of light) automatically based on the user's operating system setting (#979 by @hoechenberger)
- Bundle all previously-external JavaScript to better preserve users' privacy (#982 by @hoechenberger)
- Document the need for offscreen rendering support when running on headless servers (#997 by @drammock)

### :medical_symbol: Code health and infrastructure

- Switch from using relative to using absolute imports (#969 by @hoechenberger)
- Enable strict type checking via mypy (#995, #1013, #1016 by @larsoner)
- Improve logging messages in maxwell filtering steps. (#893 by @drammock)
- Validate extra config params passed in during testing. (#1044 by @drammock)
- New testing/example dataset "funloc" added. (#1045 by @drammock)
- Better testing of session-specific MRIs. (#1039 by @drammock)
