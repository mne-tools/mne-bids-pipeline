---
authors:
  MerlinDemeur: "[Merlin Dumeur](https://github.com/MerlinDumeur)"
  agramfort: "[Alex Gramfort](https://github.com/agramfort)"
  hoechenberger: "[Richard Höchenberger](https://github.com/hoechenberger)"
---


## Changes since April 15, 2021

### New features & enhancements

- ...

### Bug fixes

- The FreeSurfer script could only be run if `--n_jobs` was passed explicitly
  ({{ gh(287) }} by {{ authors.MerlinDemeur }})
- Fix a problem with the FreeSurfer processing step that caused the error
  message `Could not consume arg` after completion ({{ gh(301) }} by
  {{ authors.hoechenberger }})
