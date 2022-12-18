---
tags:
  - preprocessing
  - artifact-removal
  - raw
  - epochs
---

When using electric stimulation systems, e.g. for median nerve or index
stimulation, it is frequent to have a stimulation artifact. This option
allows to fix it by linear interpolation early in the pipeline on the raw
data.

::: mne_bids_pipeline._config
    options:
      members:
        - fix_stim_artifact
        - stim_artifact_tmin
        - stim_artifact_tmax
