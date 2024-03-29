## v1.8.0 (2024-03-20)

### :new: New features & enhancements

- Disabling CSP time-frequency mode is now supported by passing an empty list to [`decoding_csp_times`][mne_bids_pipeline._config.decoding_csp_times] (#890 by @whoever)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- MNE-BIDS-Pipeline now explicitly depends on `annotated-types` (#886 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- Fix handling of Maxwell-filtered data in CSP (#890 by @larsoner)
- Avoid recomputation / cache miss when the same empty-room file is matched to multiple subjects (#890 by @larsoner)

### :medical_symbol: Code health

- We removed the unused settings `shortest_event` and `study_name`. They were relics of early days of the pipeline
  and haven't been in use for a long time. (#888, #889 by @hoechenberger and @larsoner)

[//]: # (- Whatever (#000 by @whoever))
