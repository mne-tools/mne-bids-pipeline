## v1.9.0

### :new: New features & enhancements

- Added number of subject to `sub-average` report (#902 by @SophieHerbst)
- The type annotations in the default configuration file are now easier to read: We
  replaced `Union[X, Y]` with `X | Y` and `Optional[X]` with `X | None`. (#908 by @hoechenberger)

[//]: # (- Whatever (#000 by @whoever))

### :warning: Behavior changes

- All ICA HTML reports have been consolidated in the standard subject `*_report.html`
  file instead of producing separate files (#899 by @larsoner).

### :package: Requirements

- We dropped support for Python 3.9. You now need Python 3.10 or newer.

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :bug: Bug fixes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health and infrastructure

- Use GitHub's `dependabot` service to automatically keep GitHub Actions up-to-date. (#893 by @hoechenberger)
