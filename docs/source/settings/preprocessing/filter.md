---
tags:
  - preprocessing
  - frequency-filter
  - raw
---

It is typically better to set your filtering properties on the raw data so
as to avoid what we call border (or edge) effects.

If you use this pipeline for evoked responses, you could consider
a low-pass filter cut-off of h_freq = 40 Hz
and possibly a high-pass filter cut-off of l_freq = 1 Hz
so you would preserve only the power in the 1Hz to 40 Hz band.
Note that highpass filtering is not necessarily recommended as it can
distort waveforms of evoked components, or simply wash out any low
frequency that can may contain brain signal. It can also act as
a replacement for baseline correction in Epochs. See below.

If you use this pipeline for time-frequency analysis, a default filtering
could be a high-pass filter cut-off of l_freq = 1 Hz
a low-pass filter cut-off of h_freq = 120 Hz
so you would preserve only the power in the 1Hz to 120 Hz band.

If you need more fancy analysis, you are already likely past this kind
of tips! 😇

::: mne_bids_pipeline._config
    options:
      members:
        - l_freq
        - h_freq
        - l_trans_bandwidth
        - h_trans_bandwidth
        - notch_freq
        - notch_trans_bandwidth
        - notch_widths
