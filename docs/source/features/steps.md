Processing steps
================

The following table provides a concise summary of each step in the Study
Template. You can find the scripts in the `scripts` directory.

Preprocessing
-------------

| Script | Description | Run via `python run.py …` |
|:-----------|:---------------------|:--------------------------|
| `01-import_and_maxfilter.py` | Import raw data and apply Maxwell filter. | `preprocessing/import_and_maxfilter` |
| `02-frequency_filter.py` | Apply low- and high-pass filters. | `preprocessing/frequency_filter` |
| `03-make_epochs.py` | Extract epochs. | `preprocessing/make_epochs` |
| `04a-run_ica.py` | Run Independant Component Analysis (ICA) for artifact correction. | `preprocessing/run_ica` |
| `04b-run_ssp.py` | Run Signal Subspace Projections (SSP) for artifact correction. These are often also referred to as PCA vectors. | `preprocessing/run_ssp` |
| `05a-apply_ica.py` | As an alternative to ICA, you can use SSP projections to correct for eye blink and heart beat artifacts. Use either 5a/6a, or 5b/6b. | `preprocessing/apply_ica` |
| `05b-apply_ssp.py` | Apply SSP projections and obtain the cleaned epochs.  | `preprocessing/apply_ssp` |
| — | Run all preprocessing scripts.  | `preprocessing` |

Sensor-level analysis
---------------------

| Script | Description | Run via `python run.py …` |
|:-----------|:---------------------|:--------------------------|
| `01-make_evoked.py` | Extract evoked data for each condition. | `sensor/make_evoked` |
| `02-sliding_estimator.py` | Running a time-by-time decoder with sliding window. | `sensor/sliding_estimator` |
| `03-time_frequency.py` | Running a time-frequency analysis. | `sensor/time_frequency` |
| `04-group_average_sensors.py` | Make a group average of the time domain data. | `sensor/group_average_sensors` |
| — | Run all sensor-level analysis scripts. | `sensor` |

Source-level analysis
---------------------

| Script | Description | Run via `python run.py …` |
|:-----------|:---------------------|:--------------------------|
| `01-make_forward.py` | Compute forward operators. You will need to have computed the coregistration to obtain the `-trans.fif` files for each subject. | `source/make_forward` |
| `02-make_cov.py` | Compute noise covariances for each subject. | `source/make_cov` |
| `03-make_inverse.py` | Compute inverse problem to obtain source estimates. | `source/make_inverse` |
| `04-group_average_source.py` | Compute source estimates average over subjects. | `source/group_average_source` |
| — | Run all source-level analysis scripts. | `source` |

Analysis reports
----------------

| Script | Description | Run via `python run.py …` |
|:-----------|:---------------------|:--------------------------|
| `01-make_reports.py` | Compute HTML reports for each subject. | `report/make_reports`
| — | Run all report-generating scripts (currently only one). | `report`
