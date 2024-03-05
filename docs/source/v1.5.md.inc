## v1.5.0 (2023-11-30)

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
- Reduced logging when reports are created and saved (#799 by @hoechenberger)
- Added [`"picard-extended_infomax"`][mne_bids_pipeline._config.ica_algorithm] ICA algorithm to perform "extended Infomax"-like ICA decomposition using Picard (#801 by @hoechenberger)
- Added support for using "local" [`autoreject`](https://autoreject.github.io) to find (and repair) bad channels on a
  per-epoch basis as the last preprocessing step; this can be enabled by setting [`reject`][mne_bids_pipeline._config.reject]
  to `"autoreject_local"`. The behavior can further be controlled via the new setting
  [`autoreject_n_interpolate`][mne_bids_pipeline._config.autoreject_n_interpolate]. (#807 by @hoechenberger)
- Added support for "local" [`autoreject`](https://autoreject.github.io) to remove bad epochs
  before submitting the data to ICA fitting. This can be enabled by setting [`ica_reject`][mne_bids_pipeline._config.ica_reject]
  to `"autoreject_local"`. (#810, #816 by @hoechenberger)
- The new setting [`decoding_which_epochs`][mne_bids_pipeline._config.decoding_which_epochs] controls which epochs (e.g., uncleaned, after ICA/SSP, cleaned) shall be used for decoding. (#819 by @hoechenberger)
- Website documentation tables can now be sorted (e.g., to find examples that use a specific feature) (#808 by @larsoner)

### :warning: Behavior changes

- The default cache directory is now `_cache` within the derivatives folder when using `memory_location=True`, set [`memory_subdir="joblib"`][mne_bids_pipeline._config.memory_subdir] to get the behavior from v1.4 (#778 by @larsoner)
- Before cleaning epochs via ICA, we used to reject any epochs execeeding the [`ica_reject`][mne_bids_pipeline._config.ica_reject]
  criteria. However, this may lead to the unnecessary exclusion of epochs that could have been salvaged through ICA cleaning. Now,
  we only apply `ica_reject` to the epochs used for ICA fitting. After the experimental epochs have been cleaned with ICA
  (`preprocessing/apply_ica` step), any remaining large-amplitude artifacts can be removed via
  [`reject`][mne_bids_pipeline._config.reject], which is used in the last preprocessing step, `preprocessing/ptp_reject`. (#806 by @hoechenberger)
- MVPA / decoding used to be performed on un-cleaned epochs in the past. Now, cleaned epochs will be used by default (please also see the "Bug fixes" section below). (#796 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Fixed doc build errors and dependency specifications (#755 by @larsoner)
- Ensure `memory_file_method = "hash"` is tested (#768 by @larsoner)
- Enable [pre-commit.ci](https://pre-commit.ci) (#774 by @larsoner)
- Use `pooch` for web downloads (#775 by @larsoner)
- Ensure compatibility with MNE-Python 1.6 (#800 by @hoechenberger)
- Updated testing dataset for ds004229 v1.0.3 (#808 by @larsoner)

### :bug: Bug fixes

- Fixed bug where cache would not invalidate properly based on output file changes and steps could be incorrectly skipped. All steps will automatically rerun to accommodate the new, safer caching scheme (#756 by @larsoner)
- Fixed bug with parallelization across runs for Maxwell filtering (#761 by @larsoner)
- Fixed bug where head position files were not written with a proper suffix and extension (#761 by @larsoner)
- Fixed bug where default values for `decoding_csp_times` and `decoding_csp_freqs` were not set dynamically based on the config parameters (#779 by @larsoner)
- Fixed bug where the MNE logger verbosity was not respected inside parallel jobs (#813 by @larsoner)
- A number of processing steps erroneously **always** operated on un-cleaned epochs (`sensor/decoding_full_epochs`, `sensor/decoding_time_by_time`, `sensor/decoding_csp`); or operated on un-cleaned epochs (without PTP rejection) if no ICA or SSP was requested (`sensor/ime_frequency`, `sensor/make_cov`) The bug in `sensor/make_cov` could propagate to the source level, as the covariance matrix is used for inverse modeling. (#796 by @hoechenberger)
- Bad channels may have been submitted to MVPA (full epochs decoding, time-by-time decoding, CSP-based decoding) when not using Maxwell filtering
  (i.e., usually only EEG data was affected). This has now been fixed and data from bad channels is omitted from decoding. (#817 by @hoechenberger)
