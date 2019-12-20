[![CircleCI](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline.svg?style=svg)](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline)

# MNE-study-template

This repository contains an exemplary pipeline for processing MEG/EEG data
using [MNE-Python](mne.tools) and the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/).

The study template expects input data to adhere to BIDS. You can check whether
your data complies with the BIDS standard using the [BIDS validator]().

# Installation

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](http://martinos.org/mne/stable/install_mne_python.html).
Once this is done, you should be able to run this in a terminal:

`$ python -c "import mne; mne.sys_info()"`

You can then install the following additional packages via `pip`. Note that
the URL points to the bleeding edge version of `mne_bids`:

`$ pip install https://github.com/mne-tools/mne-bids/zipball/master`

# Usage

## Sample Data
You can run the template on the mne sample subject, which you need to convert to BIDS [as described here.](https://mne.tools/mne-bids/auto_examples/convert_mne_sample.html)

Another option is to fetch the data, see the [section on Contributing.](https://github.com/mne-tools/mne-study-template/blob/master/CONTRIBUTING.md)

## General

Generally, there is a single `config.py` file, which contains all parameters
for the analysis of the data. Many parameters are automatically inferred from
the BIDS structure of the data.

All other scripts should not be edited.

## Makefile

To ease interaction with the study template, there is a `Makefile`. Simply
type `make` from the root of your study template to see a summary of what
you can do, or inspect the file directly.

For Windows users, it might be necessary to install [GNU make](https://chocolatey.org/packages/make).

## Running on your own data

1. Make sure your data is formatted in BIDS
1. Set an environment variable `BIDS_ROOT` to point to your dataset
1. (optional) Set an environment variable `MNE_BIDS_STUDY_CONFIG` to point to
   a custom `config_<dataset_name>.py` file that you created to overwrite
	 the standard parameters in the main `config.py` file.
1. Use the `Makefile` to run your analyses

# Processing steps

The following table provides a concise summary of each step in the pipeline.

| Script | Description |
|:-----------|:----------------------------------------------------------|
| [config.py](config.py) | The only file you need to modify in principle. This file contain all your parameters. |
| [01-import_and_maxfilter.py](01-import_and_maxfilter.py) | Import raw data and apply Maxwell filter. |
| [02-frequency_filter.py](02-frequency_filter.py) | Apply low- and high-pass filters. |
| [03-extract_events.py](03-extract_events.py) | Extract events or annotations or markers from the data and save it to disk. Uses events from stimulus channel STI101. |
| [04-make_epochs.py](04-make_epochs.py) | Extract epochs. |
| [05a-run_ica.py](05a-run_ica.py) | Run Independant Component Analysis (ICA) for artifact correction. |
| [05b-run_ssp.py](05a-run_ssp.py) | Run Signal Subspace Projections (SSP) for artifact correction. These are often also referred to as PCA vectors. |
| [06a-apply_ica.py](06a-apply_ica.py) | As an alternative to ICA, you can use SSP projections to correct for eye blink and heart artifacts. Use either 5a/6a, or 5b/6b. |
| [06b-apply_ssp.py](06b-apply_ssp.py) | Apply SSP projections and obtain the cleaned epochs.  |
| [07-make_evoked.py](07-make_evoked.py) | Extract evoked data for each condition. |
| [08-group_average_sensors.py](08-group_average_sensors.py) | Make a group average of the time domain data. |
| [09-sliding_estimator.py](09-sliding_estimator.py) | Running a time-by-time decoder with sliding window. |
| [10-time_frequency.py](10-time_frequency.py) | Running a time-frequency analysis. |
| [11-make_forward.py](11-make_forward.py) | Compute forward operators. You will need to have computed the coregistration to obtain the `-trans.fif` files for each subject. |
| [12-make_cov.py](12-make_cov.py) | Compute noise covariances for each subject. |
| [13-make_inverse.py](13-make_inverse.py) | Compute inverse problem to obtain source estimates. |
| [14-group_average_source.py](14-group_average_source.py) | Compute source estimates average over subjects. |
| [99-make_reports.py](99-make_reports.py) | Compute HTML reports for each subject. |


# Acknowledgments

The original pipeline for MEG/EEG data processing with MNE python was build
jointly by the [Cognition and Brain Dynamics Team](https://brainthemind.com/)
and the [MNE Python Team](https://martinos.org/mne/stable/index.html),
based on scripts originally developed for this publication:

> M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen,
> A. Gramfort (2018). A reproducible MEG/EEG group study with the MNE software:
> recommendations, quality assessments, and good practices. Frontiers in
> neuroscience, 12. https://doi.org/10.3389/fnins.2018.00530

The current iteration is based on BIDS and relies on the extensions to BIDS
for EEG and MEG. See the following two references:

> Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
> Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension
> to the brain imaging data structure for electroencephalography. Scientific
> Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8

> Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
> Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
> Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
> structure extended to magnetoencephalography. Scientific Data, 5, 180110.
> https://doi.org/10.1038/sdata.2018.110
