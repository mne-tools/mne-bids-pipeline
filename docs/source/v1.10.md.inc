## v1.10.0 (unreleased)

### :new: New features & enhancements

- It is now possible to use separate MRIs for each session within a subject, as in longitudinal studies. This is achieved by creating separate "subject" folders for each subject-session combination, with the naming convention `sub-XXX_ses-YYY`, in the freesurfer `SUBJECTS_DIR`. (#987 by @drammock)

[//]: # (### :warning: Behavior changes)

[//]: # (- Whatever (#000 by @whoever))

[//]: # (### :package: Requirements)

[//]: # (- Whatever (#000 by @whoever))

### :bug: Bug fixes

- Empty room matching is now done for all sessions (previously only for the first session) for each subject. (#976 by @drammock)

- When running the pipeline with [`mf_filter_chpi`][mne_bids_pipeline._config.mf_filter_chpi] enabled (#977 by @drammock and @larsoner):

    1. Emptyroom files that lack cHPI channels will now be processed (for line noise only) instead of raising an error.
    2. cHPI filtering is now performed before movement compensation.

### :books: Documentation

- Choose the theme (dark of light) automatically based on the user's operating system setting (#979 by @hoechenberger)
- Bundle all previously-external JavaScript to better preserve users' privacy (#982 by @hoechenberger)

### :medical_symbol: Code health

- Switch from using relative to using absolute imports (#969 by @hoechenberger)

### :medical_symbol: Code health and infrastructure

- Improve logging messages in maxwell filtering steps. (#893 by @drammock)
