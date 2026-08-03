# Dev version

## v1.11.0 (unreleased)

### :new: New features & enhancements

- Added [`ignore_warnings`][mne_bids_pipeline._config.ignore_warnings] config option to allow users to specify warnings to ignore when calling `read_raw_bids` (#1224 by @larsoner)
- Added tracking of the current step in the terminal title (#1266 by @larsoner)

### :warning: Behavior changes

- The default for [`mne_log_level`][mne_bids_pipeline._config.mne_log_level] has been changed from `'error'` to `'warning'` to make possible dataset and processing errors more visible (#1224 by @larsoner)

[//3]: # (### :package: Requirements)

### :bug: Bug fixes

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
