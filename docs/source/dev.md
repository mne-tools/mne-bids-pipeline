# Dev version

## v1.11.0 (unreleased)

### :new: New features & enhancements

- Added [`ignore_warnings`][mne_bids_pipeline._config.ignore_warnings] config option to allow users to specify warnings to ignore when calling `read_raw_bids` (#1224 by @larsoner)
- Added tracking of the current step in the terminal title (#1266 by @larsoner)
- Added a "Pipeline flow" section to the reports with an auto-generated diagram of the steps that ran for a subject and the files they passed to one another (#1291 by @larsoner)
- Added a new denoising technique, [`oversampled temporal projection`][mne_bids_pipeline.steps.preprocessing._01b_otp], to the preprocessing steps (#1297 by @nordme and @sylvchev)

### :warning: Behavior changes

- The default for [`mne_log_level`][mne_bids_pipeline._config.mne_log_level] has been changed from `'error'` to `'warning'` to make possible dataset and processing errors more visible (#1224 by @larsoner)

### :package: Requirements

- Minimum supported versions were raised to Python 3.11, MNE-Python 1.8, MNE-BIDS 0.16, and joblib 1.4.1, following [SPEC 0](https://scientific-python.org/specs/spec-0000) (#1289 by @larsoner)

### :bug: Bug fixes

- Fixed report section ordering: run-specific sections now always appear in run order, even when parallelization across runs finishes them out of order (#1293 by @larsoner)
- Handle contrasts with too few epochs for cross-validation by saving NaN scores, excluding invalid subject-level results from group statistics, and reporting the effective sample size (#1265 by @viranovskaya)
- Raise an informative error if [`rest_epochs_duration`][mne_bids_pipeline._config.rest_epochs_duration] is not set for resting-state data and document the parameter (#1272 by @viranovskaya)
- Fixed bug where [`log_level`][mne_bids_pipeline._config.log_level] was not being applied to the MBPlogger (#1224 by @larsoner)
- Corrected import order: remove channels before setting template montage as stated in [`eeg_template_montage`][mne_bids_pipeline._config.eeg_template_montage] (#1220 by @dnacombo)
- Fixed crash when concatenating epochs from runs with different bad channels. The pipeline now uses the union of bad channels across runs. (#1242 by @hoechenberger)
- Fixed a small CSP labeling glitch in the report. (#1241 by @hoechenberger)
- Fixed bug where [`on_error`][mne_bids_pipeline._config.on_error] `"continue"` and `"debug"` were not respected by the `init` and `freesurfer/recon_all` steps, which could abort the whole run instead of moving on to (or debugging) the next subject (#1022 by @larsoner)
- The pipeline now recognizes source space files named with a dash between the spacing letters and number (e.g., `sample-oct-6-src.fif`), avoiding needless recomputation (#1047 by @larsoner)

[//5]: # (### :books: Documentation)

### :medical_symbol: Code health and infrastructure

- Pinned Python version for development to 3.13. (#1243 by @hoechenberger)
- Improved the accounting of options used in each step (#1268 by @larsoner)
