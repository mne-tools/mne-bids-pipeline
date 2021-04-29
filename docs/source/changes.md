---
authors:
  MerlinDumeur: "[Merlin Dumeur](https://github.com/MerlinDumeur)"
  agramfort: "[Alex Gramfort](https://github.com/agramfort)"
  hoechenberger: "[Richard Höchenberger](https://github.com/hoechenberger)"
  guiomar: "[Julia Guiomar Niso Galán](https://github.com/guiomar)"
---


## Changes since April 15, 2021

### New features & enhancements

- The new configuration option [`ica_reject`][config.ica_reject] allows to
  exclude epochs from the ICA fit based on peak-to-peak amplitude.

### Behavior changes

- Epochs rejection based on peak-to-peak amplitude, as controlled via the
  [`reject`][config.reject] setting, will now take place **after** ICA or SSP.
  In previous versions of the Pipeline, rejection was carried out before ICA
  and SSP. The exclude epochs from ICA fitting, use the new
  [`ica_reject`][config.ica_reject] setting.
- We don't apply SSP by default anymore.
  ({{ gh(315) }} by {{ authors.hoechenberger }})
- The `use_ssp` and `use_ica` settings have been removed. Please use the
  [`spatial_filter`][config.spatial_filter] setting instead.
  ({{ gh(315) }} by {{ authors.hoechenberger }})

### Bug fixes

- The FreeSurfer script could only be run if `--n_jobs` was passed explicitly
  ({{ gh(287) }} by {{ authors.MerlinDumeur }})
- Fix a problem with the FreeSurfer processing step that caused the error
  message `Could not consume arg` after completion ({{ gh(301) }} by
  {{ authors.hoechenberger }})
- Selecting the `extended_infomax` ICA algorithm caused a crash
  ({{ gh(308) }} by {{ authors.hoechenberger }})
- Correctly handle `eog_channels = None` setting after creation of bipolar EEG
  channels
  ({{ gh(311) }} by {{ authors.hoechenberger }})
- Added instructions on how to handle `FileNotFoundError` when loading the BEM
  model in the source steps ({{ gh(304) }}  by {{ authors.MerlinDumeur }})
