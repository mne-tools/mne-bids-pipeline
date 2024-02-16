[//]: # (Don't forget to add this to changes.md as an include!)

## vX.Y.0 (unreleased)

:new: New features & enhancements

- Added [`regress_artifact`][mne_bids_pipeline._config.regress_artifact] to allow artifact regression (e.g., of MEG reference sensors in KIT systems) (#837 by @larsoner)
- Chosen `reject` parameters are now saved in the generated HTML reports (#839 by @larsoner)
- Added saving of clean raw data in addition to epochs (#840 by @larsoner)
- Added saving of detected blink and cardiac events used to calculate SSP projectors (#840 by @larsoner)
- Improved event table formatting in the HTML reports (#853 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- MNE-BIDS-Pipeline now requires Python 3.9 or newer. (#825 by @hoechenberger)

### :bug: Bug fixes

- Fix minor issues with path handling for cross-talk and calibration files (#834 by @larsoner)
- Fix bug where EEG `reject` params were not used for `ch_types = ["meg", "eeg"]` (#839 by @larsoner)
- Fix bug where implicit `mf_reference_run` could change across invocations of `mne_bids_pipeline`, breaking caching (#839 by @larsoner)
- Fix bug where `--no-cache` had no effect (#839 by @larsoner)
- Fix bug where the Maxwell filtering step would fail if [`find_noisy_channels_meg = False`][mne_bids_pipeline._config.find_noisy_channels_meg]` was used (#847 by @larsoner)
- Fix bug where raw, empty-room, and custom noise covariances were errantly calculated on data without ICA or SSP applied (#840 by @larsoner)
- Fix bug where multiple channel types (e.g., MEG and EEG) were not handled correctly in decoding (#853 by @larsoner)
- Fix bug where the previous default for [`ica_n_components`][mne_bids_pipeline._config.ica_n_components] of `0.8` was too conservative, changed the default to `None` to match MNE-Python (#853 by @larsoner)

### :medical_symbol: Code health

- The package build backend has been switched from `setuptools` to `hatchling`. (#825 by @hoechenberger)
- Code formatting now uses `ruff format` instead of `black` (#834, #838 by @larsoner)
- Code caching is now tested using GitHub Actions (#836 by @larsoner)
