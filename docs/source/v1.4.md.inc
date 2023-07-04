## v1.4.0 (2023-07-04)

### :new: New features & enhancements

- Add movement compensation and cHPI filtering to the Maxwell filtering step, along with additional configuration options (#747 by @larsoner)
- Add option to specify [`ssp_ecg_channel`]([mne_bids_pipeline._config.ssp_ecg_channel) to override the default value (#747 by @larsoner)
- Add option [`read_raw_bids_verbose`]([mne_bids_pipeline._config.read_raw_bids_verbose) to set the verbosity level when using `read_raw_bids` to suppress known warnings (#749 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Refactor code to deduplicate keyword-passing and improve internal logic (#746, #749 by @larsoner)

### :bug: Bug fixes

- Fix bug when [`mf_reference_run != runs[0]`]([mne_bids_pipeline._config.mf_reference_run) (#742 by @larsoner)
- Fix bug with too many JSON files found during empty-room discovery (#743 by @allermat)
- Fix bug where SSP projectors were not added to the report (#747 by @larsoner)
- Fix documentation of `data_type` configuration option (#751 by @allermat)
- Fix documentation of `ch_types` configuration option (#745 by @allermat)
