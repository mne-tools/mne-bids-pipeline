## v1.9.0

### :new: New features & enhancements

- Added number of subject to `sub-average` report (#902 by @SophieHerbst)
- The type annotations in the default configuration file are now easier to read: We
  replaced `Union[X, Y]` with `X | Y` and `Optional[X]` with `X | None`. (#908 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- We dropped support for Python 3.9. You now need Python 3.10 or newer.

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- Fixed incorrect inclusion of autobad channels when disabling `find_bad_channels_meg` for example after enabling it (#902 by @larsoner)
- Fixed group-average decoding statistics were not updated in some cases, even if relevant configuration options had been changed. (#902 by @larsoner)

### :medical_symbol: Code health and infrastructure

- Use GitHub's `dependabot` service to automatically keep GitHub Actions up-to-date. (#893 by @hoechenberger)
