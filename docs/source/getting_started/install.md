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

??? info "Detailed list of dependencies"
    - `mne-bids` to operate on BIDS data
    - `autoreject` to automatically detect problematic epochs based on
      peak-to-peak (PTP) amplitudes
    - `coloredlogs` for nicer logging output
    - `tqdm` for progress bars
    - `pandas` for table creation
    - `openpyxl` to write Excel files used for logging
    - `json_tricks` for handling of some analysis output
    - `scikit-learn` for decoding
    - `fire` for the command line interface
    - `typing_extensions` if you're using a Python version older than 3.8

Download MNE-BIDS-Pipeline
--------------------------

Download the [code](https://github.com/mne-tools/mne-bids-pipeline/archive/refs/heads/main.zip)
