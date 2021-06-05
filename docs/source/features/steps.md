Processing steps
================

The following table provides a concise summary of each step in the Study
Template. You can find the scripts in the `scripts` directory.

Preprocessing
-------------

| Processing step                      | Description |
|:-------------------------------------|:------------|
| `preprocessing`                      | Run all preprocessing scripts. |
| `preprocessing/import_and_maxfilter` | Import raw data and apply Maxwell filter. |
| `preprocessing/frequency_filter`     | Apply low- and high-pass filters. |
| `preprocessing/make_epochs`          | Extract epochs. |
| `preprocessing/run_ica`              | Run Independant Component Analysis (ICA) for artifact correction. |
| `preprocessing/run_ssp`              | Run Signal Subspace Projections (SSP) for artifact correction. These are often also referred to as PCA vectors. |
| `preprocessing/apply_ica`            | As an alternative to ICA, you can use SSP projections to correct for eye blink and heart beat artifacts. |
| `preprocessing/apply_ssp`            | Apply SSP projections and obtain the cleaned epochs. |

Sensor-level analysis
---------------------

| Processing step                | Description |
|:-------------------------------|:------------|
| `sensor`                       | Run all sensor-level analysis scripts. |
| `sensor/make_evoked`           | Extract evoked data for each condition. |
| `sensor/sliding_estimator`     | Running a time-by-time decoder with sliding window. This is achieved by standardising the features, applying a logistic-regression classifier, then applying the sliding estimator. The returned scores are generated using a stratified k-folds validator. The subject-level report presents the decoding performance across epochs. The average-subject report presents the mean performance across subjects. |
| `sensor/time_frequency`        | Running a time-frequency analysis. |
| `sensor/group_average_sensors` | Make a group average of the time domain data. |

Source-level analysis
---------------------

| Processing step        | Description |
|:-----------------------|:------------|
| `source`               | Run all source-level analysis scripts. |
| `source/make_forward`  | Compute forward operators. You will need to have computed the coregistration to obtain the `-trans.fif` files for each subject. |
| `source/make_cov`      | Compute noise covariances for each subject. |
| `source/make_inverse`  | Compute inverse solution to obtain source estimates. |
| `source/group_average` | Compute source estimates average over subjects. |

Analysis reports
----------------

| Processing step          | Description |
|:-------------------------|:------------|
| `report`                 | Run all report-generating scripts (currently only one). |
| `report/make_reports.py` | Compute HTML reports for each subject. |
