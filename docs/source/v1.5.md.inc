## v1.5.0 (unreleased)

This release contains a number of very important bug fixes that address problems related to decoding, time-frequency analysis, and inverse modeling.
All users are encouraged to update.

### :new: New features & enhancements

- Added `deriv_root` argument to CLI (#773 by @vferat)
- Added support for annotating bad segments based on head movement velocity (#757 by @larsoner)
- Added examples of T1 and FLASH BEM to website (#758 by @larsoner)
- Added support for extended SSS (eSSS) in Maxwell filtering (#762 by @larsoner)
- Output logging spacing improved (#764 by @larsoner)
- Added caching of sensor and source average steps (#765 by @larsoner)
- Improved logging of coregistration distances (#769 by @larsoner)
- Input validation has been improved by leveraging [pydantic](https://docs.pydantic.dev) (#779 by @larsoner)

[//]: # (### :warning: Behavior changes)

- The default cache directory is now `_cache` within the derivatives folder when using `memory_location=True`, set [`memory_subdir="joblib"`][mne_bids_pipeline._config.memory_subdir] to get the behavior from v1.4 (#778 by @larsoner)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Fixed doc build errors and dependency specifications (#755 by @larsoner)
- Ensure `memory_file_method = "hash"` is tested (#768 by @larsoner)
- Enable [pre-commit.ci](https://pre-commit.ci) (#774 by @larsoner)
- Use `pooch` for web downloads (#775 by @larsoner)

### :bug: Bug fixes

- Fixed bug where cache would not invalidate properly based on output file changes and steps could be incorrectly skipped. All steps will automatically rerun to accommodate the new, safer caching scheme (#756 by @larsoner)
- Fixed bug with parallelization across runs for Maxwell filtering (#761 by @larsoner)
- Fixed bug where head position files were not written with a proper suffix and extension (#761 by @larsoner)
- Fixed bug where default values for `decoding_csp_times` and `decoding_csp_freqs` were not set dynamically based on the config parameters (#779 by @larsoner)
- A number of processing steps erroneously **always** operated on un-cleaned epochs (`sensor/decoding_full_epochs`, `sensor/decoding_time_by_time`, `sensor/decoding_csp`); or operated on un-cleaned epochs (without PTP rejection) if no ICA or SSP was requested (`sensor/ime_frequency`, `sensor/make_cov`) The bug in `sensor/make_cov` could propagate to the source level, as the covariance matrix is used for inverse modeling. (#796 by @hoechenberger)