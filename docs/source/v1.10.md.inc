## v1.10.0 (unreleased)

[//]: # (### :new: New features & enhancements)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :package: Requirements)

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- When running the pipeline with [`mf_filter_chpi`][mne_bids_pipeline._config. mf_filter_chpi] enabled,
  emptyroom files that lack cHPI channels will now be processed (for line noise only) instead of raising
  an error. (#977 by @drammock)

### :medical_symbol: Code health

- Switch from using relative to using absolute imports (#969 by @hoechenberger)
