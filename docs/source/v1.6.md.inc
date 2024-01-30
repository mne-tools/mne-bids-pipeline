[//]: # (Don't forget to add this to changes.md as an include!)

## vX.Y.0 (unreleased)

:new: New features & enhancements

- Added [`regress_artifact`][mne_bids_pipeline._config.regress_artifact] to allow artifact regression (e.g., of MEG reference sensors in KIT systems) (#837 by @larsoner)
- Chosen `reject` parameters are now saved in the generated HTML reports (#839 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- MNE-BIDS-Pipeline now requires Python 3.9 or newer. (#825 by @hoechenberger)

### :bug: Bug fixes

- Fix minor issues with path handling for cross-talk and calibration files (#834 by @larsoner)
- Fix bug where EEG `reject` params were not used for `ch_types = ["meg", "eeg"]` (#839 by @larsoner)
- Fix bug where implicit `mf_reference_run` could change across invocations of `mne_bids_pipeline`, breaking caching (#839 by @larsoner)

### :medical_symbol: Code health

- The package build backend has been switched from `setuptools` to `hatchling`. (#825 by @hoechenberger)
- Code formatting now uses `ruff format` instead of `black` (#834, #838 by @larsoner)
- Code caching is now tested using GitHub Actions (#836 by @larsoner)
