Install MNE-Python
------------------

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install/mne_python.html).

Install additional dependencies
-------------------------------

You will also need to install a number of additional dependencies that are
required to run the pipeline.

Run in your terminal:

```shell
pip install -r https://raw.githubusercontent.com/mne-tools/mne-bids-pipeline/main/requirements.txt
```

??? info "Detailed list of additional dependencies"
    - `mne-bids[full]` to operate on BIDS data with all dependencies
    - `autoreject` to automatically detect problematic epochs based on
      peak-to-peak (PTP) amplitudes
    - `coloredlogs` for nicer logging output
    - `pandas` for table creation
    - `seaborn` for certain plots
    - `openpyxl` to write Excel files used for logging
    - `json_tricks` for handling of some analysis output
    - `scikit-learn` for decoding
    - `fire` for the command line interface
    - `typing_extensions` if you're using a Python version older than 3.8
    - `pyvista` for producing 3D brain plots in reports with source space results.
    - `python-picard` for using Picard as ICA method
    - `dask[distributed]` for parallel processing using [Dask](https://dask.org)

Now that everything is installed
--------------------------------

[Download and extract the code :fontawesome-solid-floppy-disk:](https://github.com/mne-tools/mne-bids-pipeline/archive/refs/heads/main.zip){: .md-button .md-button--primary }

[Discover Basic Usage :fontawesome-solid-rocket:](basic_usage.html){: .md-button .md-button--primary }
