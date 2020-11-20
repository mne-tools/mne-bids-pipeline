Processing steps
================

The following table provides a concise summary of each step in the Study
Template. You can find the scripts in the `scripts` directory.

| Group | Script | Description |
|:-----------|:-----------|:-----------------------------------------------|
| `preprocessing` | `01-import_and_maxfilter.py` | Import raw data and apply Maxwell filter. |
| `preprocessing` | `02-frequency_filter.py` | Apply low- and high-pass filters. |
| `preprocessing` | `03-make_epochs.py` | Extract epochs. |
| `preprocessing` | `04a-run_ica.py` | Run Independant Component Analysis (ICA) for artifact correction. |
| `preprocessing` | `04b-run_ssp.py` | Run Signal Subspace Projections (SSP) for artifact correction. These are often also referred to as PCA vectors. |
| `preprocessing` | `05a-apply_ica.py` | As an alternative to ICA, you can use SSP projections to correct for eye blink and heart beat artifacts. Use either 5a/6a, or 5b/6b. |
| `preprocessing` | `05b-apply_ssp.py` | Apply SSP projections and obtain the cleaned epochs.  |
| `sensor` | `06-make_evoked.py` | Extract evoked data for each condition. |
| `sensor` | `07-sliding_estimator.py` | Running a time-by-time decoder with sliding window. |
| `sensor` | `08-time_frequency.py` | Running a time-frequency analysis. |
| `sensor` | `09-group_average_sensors.py` | Make a group average of the time domain data. |
| `source` | `10-make_forward.py` | Compute forward operators. You will need to have computed the coregistration to obtain the `-trans.fif` files for each subject. |
| `source` | `11-make_cov.py` | Compute noise covariances for each subject. |
| `source` | `12-make_inverse.py` | Compute inverse problem to obtain source estimates. |
| `source` | `13-group_average_source.py` | Compute source estimates average over subjects. |
| `report` | `99-make_reports.py` | Compute HTML reports for each subject. |
