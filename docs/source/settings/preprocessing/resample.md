---
tags:
  - preprocessing
  - resampling
  - decimation
  - raw
  - epochs
---

If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
you will likely want to downsample to lighten up the size of the files you
are working with (pragmatics)
If you are interested in typical analysis (up to 120 Hz) you can typically
resample your data down to 500 Hz without preventing reliable time-frequency
exploration of your data.

::: mne_bids_pipeline._config
    options:
      members:
        - raw_resample_sfreq
        - epochs_decim
