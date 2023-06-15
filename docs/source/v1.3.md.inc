## v1.3.0 (released 2023/06/01)

### :new: New features & enhancements

- Provide a more helpful log message if the CSP decoding step is being skipped (#734 by @hoechenberger)
- Use `rich` for improved logging control and styling (#737 by @larsoner)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Avoid using deprecated `openpyxl` API when working with Excel spreadsheets (#735 by @larsoner)

### :bug: Bug fixes

- Fix pandas 2.0 compatibility (#732 by @larsoner)
- Fix bug with `mne.sys_info` insertion in reports (#732 by @larsoner)
- Always generate CSP decdoing grand average analysis if CSP decoding was used on the single-subject level (#733 by @hoechenberger)
