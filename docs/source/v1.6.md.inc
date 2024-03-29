## v1.6.0 (2024-03-01)

:new: New features & enhancements

- Added [`regress_artifact`][mne_bids_pipeline._config.regress_artifact] to allow artifact regression (e.g., of MEG reference sensors in KIT systems) (#837 by @larsoner)
- Chosen `reject` parameters are now saved in the generated HTML reports (#839 by @larsoner)
- Added saving of clean raw data in addition to epochs (#840 by @larsoner)
- Added saving of detected blink and cardiac events used to calculate SSP projectors (#840 by @larsoner)
- Added [`noise_cov_method`][mne_bids_pipeline._config.noise_cov_method] to allow for the use of methods other than `"shrunk"` for noise covariance estimation (#854 by @larsoner)
- Added option to pass `image_kwargs` to [`mne.Report.add_epochs`] to allow adjusting e.g. `"vmin"` and `"vmax"` of the epochs image in the report via [`report_add_epochs_image_kwargs`][mne_bids_pipeline._config.report_add_epochs_image_kwargs]. This feature requires MNE-Python 1.7 or newer. (#848 by @SophieHerbst)
- Split ICA fitting and artifact detection into separate steps. This means that now, ICA is split into a total of three consecutive steps: fitting, artifact detection, and the actual data cleaning step ("applying ICA"). This makes it easier to experiment with different settings for artifact detection without needing to re-fit ICA. (#865 by @larsoner)
- The configuration used for the pipeline is now saved in a separate spreadsheet in the `.xlsx` log file (#869 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- MNE-BIDS-Pipeline now requires Python 3.9 or newer. (#825 by @hoechenberger)

### :bug: Bug fixes

- Fixed minor issues with path handling for cross-talk and calibration files (#834 by @larsoner)
- Fixed EEG `reject` use for `ch_types = ["meg", "eeg"]` in epoch cleaning (#839 by @larsoner)
- Fixed bug where implicit `mf_reference_run` could change across invocations of `mne_bids_pipeline`, breaking caching (#839 by @larsoner)
- Fixed `--no-cache` behavior having no effect (#839 by @larsoner)
- Fixed Maxwell filtering failures when [`find_noisy_channels_meg = False`][mne_bids_pipeline._config.find_noisy_channels_meg]` is used (#847 by @larsoner)
- Fixed raw, empty-room, and custom noise covariances calculation, previously they could errantly be calculated on data without ICA or SSP applied (#840 by @larsoner)
- Fixed multiple channel type handling (e.g., MEG and EEG) in decoding (#853 by @larsoner)
- Changed the default for [`ica_n_components`][mne_bids_pipeline._config.ica_n_components] from `0.8` (too conservative) to `None` to match MNE-Python's default (#853 by @larsoner)
- Prevent events table for the average subject overflowing in reports (#854 by @larsoner)
- Fixed split file behavior for Epochs when using ICA (#855 by @larsoner)
- Fixed a bug where users could not set `_components.tsv` as it would be detected as a cache miss and overwritten on next pipeline run (#865 by @larsoner)

### :medical_symbol: Code health

- The package build backend has been switched from `setuptools` to `hatchling`. (#825 by @hoechenberger)
- Removed dependencies on `datalad` and `git-annex` for testing (#867 by @larsoner)
- Code formatting now uses `ruff format` instead of `black` (#834, #838 by @larsoner)
- Code caching is now tested using GitHub Actions (#836 by @larsoner)
- Steps in the documentation are now automatically parsed into flowcharts (#859 by @larsoner)
- New configuration options are now automatically added to the docs (#863 by @larsoner)
- Configuration options now have relevant steps listed in the docs (#866 by @larsoner)
