## v1.2.0 (released 2023/03/23)

### :new: New features & enhancements

- We improved caching to reduce initialization time when large datasets are being processed. (#720 by @apmellot)

[//]: # (- Whatever (#000 by @whoever))

### :warning: Behavior changes

- MNE-BIDS-Pipeline now requires MNE-Python 1.2 or newer.

### :medical_symbol: Code health

- Replace legacy `plot_psd()` calls with `compute_psd().plot()`
  (#725 by @drammock)

[//]: # (### :bug: Bug fixes)
