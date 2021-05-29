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
- An official [project governance](governance.md) structure has officially
  been adopted.

### Behavior changes

- The [`conditions`][config.conditions] setting will now be `None` by default. 
  It is a required setting so it will raise an error if left as `None`.
  ({{ gh(348) }} by {{ authors.guiomar, authors.hoechenberger }})
- Epochs rejection based on peak-to-peak amplitude, as controlled via the
  [`reject`][config.reject] setting, will now take place **after** ICA or SSP.
  In previous versions of the Pipeline, rejection was carried out before ICA
  and SSP. The exclude epochs from ICA fitting, use the new
  [`ica_reject`][config.ica_reject] setting.
- We don't apply SSP by default anymore.
  ({{ gh(315) }} by {{ authors.hoechenberger }})
- The `use_ssp` and `use_ica` settings have been removed. Please use the new
  [`spatial_filter`][config.spatial_filter] setting instead.
  ({{ gh(315) }} by {{ authors.hoechenberger }})
- The `allow_maxshield` setting has been removed. The Pipeline now
  automatically ensures that FIFF files of recordings with active
  shielding (MaxShield) can be imported. Later stages of the Pipeline will fail
  if Maxwell filtering of such data is disabled via `use_maxwell_filter=False`.
  ({{ gh(318) }} by {{ authors.hoechenberger }})
- The overlay plots that show the effects of ICA cleaning are now based on the
  baseline-corrected data to make it easier to spot the differences.
  ({{ gh(320) }} by {{ authors.hoechenberger }})
- `bids_root` and `deriv_root` are now converted to absolute paths to avoid
  running into issues caused by relative path specifications.
  ({{ gh(322) }} by {{ authors.hoechenberger }})
- Warn if using ICA and no EOG- or ECG-related ICs were detected.
  ({{ gh(351) }} by {{ authors.crsegerie }})

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
- When using [`find_noisy_channels_meg`][config.find_noisy_channels_meg] or
  [`find_flat_channels_meg`][config.find_flat_channels_meg], we now pass [`mf_head_origin`][config.mf_head_origin] to the respective bad channel
  detection algorithm to achieve better performance
  ({{ gh(319) }} by {{ authors.agramfort }})
- Baseline was not applied to epochs if neither ICA nor SSP was used
  ({{ gh(319) }} by {{ authors.hoechenberger }})
- Ensure we always use the cleaned epochs for constructing evoked data
  ({{ gh(319) }} by {{ authors.agramfort }})
- The summary report didn't use the cleaned epochs for showing the effects of
  ICA.
  ({{ gh(320) }} by {{ authors.hoechenberger }})
- The sanity check comparing the rank of the experimental data and the rank of the empty-room after Maxwell-filtering did not use the maxfiltered data.
  ({{ gh(336) }} by {{ authors.agramfort }}, {{ authors.hoechenberger }}, and {{ authors.crsegerie }})
- `epochs_tmin` and `epochs_tmax` were named incorrectly in some test config files.
  ({{ gh(340) }} by {{ authors.crsegerie }})
