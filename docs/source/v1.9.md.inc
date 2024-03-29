## v1.9.0

### :new: New features & enhancements

- Added number of subject to `sub-average` report (#902, #910 by @SophieHerbst)
- The type annotations in the default configuration file are now easier to read: We
  replaced `Union[X, Y]` with `X | Y` and `Optional[X]` with `X | None`. (#908 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- We dropped support for Python 3.9. You now need Python 3.10 or newer.

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- When running the pipeline with [`find_bad_channels_meg`][mne_bids_pipeline._config. find_bad_channels_meg] enabled,
  then disabling it and running the pipeline again, the pipeline would incorrectly still use automatically detected
  bad channels from the first pipeline run. Now, we ensure that the original bad channels would be used and the
  related section is removed from the report in this case. (#902 by @larsoner)
- Fixed group-average decoding statistics were not updated in some cases, even if relevant configuration options had been changed. (#902 by @larsoner)

### :medical_symbol: Code health and infrastructure

- Use GitHub's `dependabot` service to automatically keep GitHub Actions up-to-date. (#893 by @hoechenberger)
