---
tags:
  - preprocessing
  - raw
  - bad-channels
---

!!! warning
    This functionality will soon be removed from the pipeline, and
    will be integrated into MNE-BIDS.

"Bad", i.e. flat and overly noisy channels, can be automatically detected
using a procedure inspired by the commercial MaxFilter by Elekta. First,
a copy of the data is low-pass filtered at 40 Hz. Then, channels with
unusually low variability are flagged as "flat", while channels with
excessively high variability are flagged as "noisy". Flat and noisy channels
are marked as "bad" and excluded from subsequent analysis. See
:func:`mne.preprocssessing.find_bad_channels_maxwell` for more information
on this procedure. The list of bad channels detected through this procedure
will be merged with the list of bad channels already present in the dataset,
if any.

::: mne_bids_pipeline._config
    options:
      members:
        - find_flat_channels_meg
        - find_noisy_channels_meg
