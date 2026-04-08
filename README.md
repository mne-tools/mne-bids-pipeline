# <img src="https://raw.github.com/mne-tools/mne-bids-pipeline/main/docs/source/assets/mne.svg" alt="MNE Logo" height="20"> MNE-BIDS-Pipeline

<!--keep description in sync with pyproject.toml-->

<!--tagline-start-->
**MNE-BIDS-EyeTracking-Pipeline is coregisterd eye-tracking fork of MNE-BIDS-Pipeline.**

## Contributions
- Bad Channel detection using PyPrep: Jevri Hanna, Judith Schepers
- Correct common average reference: Judith Schepers
- Synchronisation of EyeLink: Judith Schepers, Jevri Hanna, Benedikt Ehinger
- Synchronisation of SMI: Okan Mazlum (not yet merged)
- IClabel: Judith Schepers, Sven Naber -> already merged upstream
- Zapline: Boschra Ariguib -> already merged upstream

- Supervision & Code review: Benedikt Ehinger


## Original readme starts here -->
* It operates on data stored according to the [Brain Imaging Data
Structure (BIDS)](https://bids.neuroimaging.io/).
* Under the hood, it uses [MNE-Python](https://mne.tools).

<!--tagline-end-->

## 💡 Basic concepts and features

<!--features-list-start-->

* 🏆 Automated processing of MEG and EEG data from raw data to inverse solutions.
* 🛠️ Configuration via a simple text file.
* 📘 Extensive processing and analysis summary reports.
* 🧑‍🤝‍🧑 Process just a single participant, or as many as several hundreds of participants – in parallel.
* 💻 Execution via an easy-to-use command-line utility.
* 🆘 Helpful error messages in case something goes wrong.
* 👣 Data processing as a sequence of standard processing steps.
* ⏩ Steps are cached to avoid unnecessary recomputation.
* ⏏️ Data can be "ejected" from the pipeline at any stage. No lock-in!
* ☁️ Runs on your laptop, on a powerful server, or on a high-performance cluster via Dask.

<!--features-list-end-->

## 📘 Installation and usage instructions

Please find the documentation at
[**mne.tools/mne-bids-pipeline**](https://mne.tools/mne-bids-pipeline).

## ❤ Acknowledgments

The original pipeline for MEG/EEG data processing with MNE-Python was built
jointly by the [Cognition and Brain Dynamics Team](https://brainthemind.com/)
and the [MNE Python Team](https://mne.tools), based on scripts originally
developed for this publication:

> M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen,
> A. Gramfort (2018). A reproducible MEG/EEG group study with the MNE software:
> recommendations, quality assessments, and good practices. Frontiers in
> neuroscience, 12. <https://doi.org/10.3389/fnins.2018.00530>

The current iteration is based on BIDS and relies on the extensions to BIDS
for EEG and MEG. See the following two references:

> Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
> Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension
> to the brain imaging data structure for electroencephalography. Scientific
> Data, 6, 103. <https://doi.org/10.1038/s41597-019-0104-8>

> Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
> Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
> Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
> structure extended to magnetoencephalography. Scientific Data, 5, 180110.
> <https://doi.org/10.1038/sdata.2018.110>

## Contributing

see <./CONTRIBUTING.md>
