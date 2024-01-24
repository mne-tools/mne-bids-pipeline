[//]: # (Don't forget to add this to changes.md as an include!)

## vX.Y.0 (unreleased)

[//]: # (### :new: New features & enhancements)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :package: Requirements

- MNE-BIDS-Pipeline now requires Python 3.9 or newer. (#825 by @hoechenberger)

### :bug: Bug fixes

- Fix minor issues with path handling for cross-talk and calibration files (#834 by @larsoner)

### :medical_symbol: Code health

- The package build backend has been switched from `setuptools` to `hatchling`. (#825 by @hoechenberger)
- Code formatting now uses `ruff format` instead of `black` (#834 by @larsoner)
- Code caching is now tested using GitHub Actions (#836 by @larsoner)
