## v1.7.0 (unreleased)

### :new: New features & enhancements

- Epochs metadata creation now supports variable time windows by specifying the names of events via [`epochs_metadata_tmin`][mne_bids_pipeline._config.epochs_metadata_tmin] and [`epochs_metadata_tmax`][mne_bids_pipeline._config.epochs_metadata_tmax]. (#873 by @hoechenberger)
- Improved logging message during cache invalidation: We now print the selected
  [`memory_file_method`][mne_bids_pipeline._config.memory_file_method] ("hash" or "mtime").
  Previously, we'd always print "hash". (#876 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- The minimum required version of MNE-Python is now 1.7.0.

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :bug: Bug fixes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- We enabled stricter linting to guarantee a consistently high code quality! (#872 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))
