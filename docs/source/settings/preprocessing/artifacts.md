---
tags:
  - preprocessing
  - artifact-removal
  - epochs
---

???+ info "Good Practice / Advice"
    Have a look at your raw data and train yourself to detect a blink, a heart
    beat and an eye movement.
    You can do a quick average of blink data and check what the amplitude looks
    like.

::: mne_bids_pipeline._config
    options:
      members:
        - reject
        - reject_tmin
        - reject_tmax
