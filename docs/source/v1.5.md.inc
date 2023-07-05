## v1.5.0 (unreleased)

[//]: # (### :new: New features & enhancements)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Fixed doc build errors and dependency specifications (#755 by @larsoner)

### :bug: Bug fixes

- Fixed bug where cache would not invalidate properly based on output file changes and steps could be incorrectly skipped. All steps will automatically rerun to accommodate the new, safer caching scheme (#756 by @larsoner)
