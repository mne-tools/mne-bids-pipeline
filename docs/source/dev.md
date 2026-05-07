# Dev version

## v1.11.0 (unreleased)

### :new: New features & enhancements

- Added [`ignore_warnings`][mne_bids_pipeline._config.ignore_warnings] config option to allow users to specify warnings to ignore when calling `read_raw_bids` (#1224 by @larsoner)

### :warning: Behavior changes

- The default for [`mne_log_level`][mne_bids_pipeline._config.mne_log_level] has been changed from `'error'` to `'warning'` to make possible dataset and processing errors more visible (#1224 by @larsoner)

[//3]: # (### :package: Requirements)

### :bug: Bug fixes

- Fixed bug where [`log_level`][mne_bids_pipeline._config.log_level] was not being applied to the MBPlogger (#1224 by @larsoner)
- Corrected import order: remove channels before setting template montage as stated in [`eeg_template_montage`][mne_bids_pipeline._config.eeg_template_montage] (#1220 by @dnacombo)

[//5]: # (### :books: Documentation)

[//6]: # (### :medical_symbol: Code health and infrastructure)
