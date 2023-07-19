## v1.5.0 (unreleased)

### :new: New features & enhancements

- Added support for annotating bad segments based on head movement velocity (#757 by @larsoner)
- Added examples of T1 and FLASH BEM to website (#758 by @larsoner)
- Added support for extended SSS (eSSS) in Maxwell filtering (#762 by @larsoner)
- Output logging spacing improved (#764 by @larsoner)
- Added caching of sensor and source average steps (#765 by @larsoner)
- Improved logging of coregistration distances (#769 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Fixed doc build errors and dependency specifications (#755 by @larsoner)
- Ensure `memory_file_method = "hash"` is tested (#768 by @larsoner)

### :bug: Bug fixes

- Fixed bug where cache would not invalidate properly based on output file changes and steps could be incorrectly skipped. All steps will automatically rerun to accommodate the new, safer caching scheme (#756 by @larsoner)
- Fixed bug with parallelization across runs for Maxwell filtering (#761 by @larsoner)
- Fixed bug where head position files were not written with a proper suffix and extension (#761 by @larsoner)
