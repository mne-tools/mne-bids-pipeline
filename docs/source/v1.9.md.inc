## v1.9.0

### :new: New features & enhancements

- Added number of subject to `sub-average` report (#902, #910 by @SophieHerbst)
- The type annotations in the default configuration file are now easier to read: We
  replaced `Union[X, Y]` with `X | Y` and `Optional[X]` with `X | None`. (#908, #911 by @hoechenberger)
- Epochs metadata creation now supports variable time windows by specifying the names of events via
  [`epochs_metadata_tmin`][mne_bids_pipeline._config.epochs_metadata_tmin] and
  [`epochs_metadata_tmax`][mne_bids_pipeline._config.epochs_metadata_tmax]. (#873 by @hoechenberger)
- If you requested processing of non-existing subjects, we will now provide a more helpful error message. (#928 by @hoechenberger)
- We improved the logging output for automnated epochs rejection and cleaning via ICA and SSP. (#936, #937 by @hoechenberger)
- ECG and EOG signals created during ICA artifact detection are now saved to disk. (#938 by @hoechenberger)

### :warning: Behavior changes

- All ICA HTML reports have been consolidated in the standard subject `*_report.html`
  file instead of producing separate files. (#899 by @larsoner)
- Changed default for `source_info_path_update` to `None`.  In `_04_make_forward.py`
  and `_05_make_inverse.py`, we retrieve the info from the file from which
  the `noise_cov` is computed. (#919 by @SophieHerbst)
- The [`depth`][mne_bids_pipeline._config.depth] parameter doesn't accept `None`
  anymore. Please use `0` instead. (#915 by @hoechenberger)
- When using automated bad channel detection, now indicate the generated `*_bads.tsv` files whether a channel
  had previously already been marked as bad in the dataset. Resulting entries in the TSV file may now look like:
  `"pre-existing (before MNE-BIDS-pipeline was run) & auto-noisy"` (previously: only `"auto-noisy"`). (#930 by @hoechenberger)
- The `ica_ctps_ecg_threshold` has been renamed to [`ica_ecg_threshold`][mne_bids_pipeline._config.ica_ecg_threshold]. (#935 by @hoechenberger)
- We changed the behavior when setting an EEG montage:
  - When applying the montage, we now also check for channel aliases (e.g. `M1 -> TP9`).
  - If the data contains a channel that is not present in the montage, we now abort with an exception (previously, we emitted a warning).
    This is to prevent silent errors. To proceed in this situation, select a different montage, or drop the respective channels via
    the [`drop_channels`][mne_bids_pipeline._config.drop_channels] configuration option. (#960 by @hoechenberger)

### :package: Requirements

- The minimum required version of MNE-Python is now 1.7.0.
- We dropped support for Python 3.9. You now need Python 3.10 or newer. (#908 by @hoechenberger)

### :book: Documentation

- We removed the `Execution` section from configuration options documentation and
  replaced it with new, more explicit sections (namely, Caching, Parallelization,
  Logging, and Error handling), and enhanced documentation. (#914 by @hoechenberger, #916 by @SophieHerbst)

### :bug: Bug fixes

- When running the pipeline with [`find_bad_channels_meg`][mne_bids_pipeline._config. find_bad_channels_meg] enabled,
  then disabling it and running the pipeline again, the pipeline would incorrectly still use automatically detected
  bad channels from the first pipeline run. Now, we ensure that the original bad channels would be used and the
  related section is removed from the report in this case. (#902 by @larsoner)
- Fixed group-average decoding statistics were not updated in some cases, even if relevant configuration options had been changed. (#902 by @larsoner)
- Fixed a compatibility bug with joblib 1.4.0. (#899 by @larsoner)
- Fixed how "original" raw data is included in the report. Previously, bad channels, subject, and experimenter name would not
  be displayed correctly. (#930 by @hoechenberger)
- In the report's table of contents, don't put the run numbers in quotation marks. (#933 by @hoechenberger)

### :medical_symbol: Code health and infrastructure

- Use GitHub's `dependabot` service to automatically keep GitHub Actions up-to-date. (#893 by @hoechenberger)
