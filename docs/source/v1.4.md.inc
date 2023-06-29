## v1.4.0 (unreleased)

### :new: New features & enhancements

- Add movement compensation and cHPI filtering to the Maxwell filtering step, along with additional configuration options (#747 by @larsoner)
- Add option to specify `ssp_ecg_channel` to override the default value (#747 by @larsoner)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

### :medical_symbol: Code health

- Refactor code to deduplicate keyword-passing (#746 by @larsoner)

### :bug: Bug fixes

- Fix bug when `mf_reference_run != runs[0]` (#742 by @larsoner)
- Fix bug with too many JSON files found during empty room matching (#743 by @allermat)
- Fix bug with outdated info on ch_types config option (#745 by @allermat)
- Fix bug where SSP projectors were not added to the report (#747 by @larsoner)
- Fix bug with documentation issue on data_type config option (#750 by @allermat)