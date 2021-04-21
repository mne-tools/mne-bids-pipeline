---
authors:
  MerlinDemeur: "[Merlin Dumeur](https://github.com/MerlinDumeur)"
  agramfort: "[Alex Gramfort](https://github.com/agramfort)"
  hoechenberger: "[Richard Höchenberger](https://github.com/hoechenberger)"
---


## Changes since April 15, 2021

### New features & enhancements

- New configuration option `reject_exclusions` to exclude specific channels
  when applying rejection thresholds to epochs ({{ gh(xxx) }} by
  {{ authors.hoechenberger}})

### Bug fixes

- The FreeSurfer script could only be run if `--n_jobs` was passed explicitly
  ({{ gh(287) }} by {{ authors.MerlinDemeur }})
