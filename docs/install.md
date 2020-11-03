1. Install MNE-Python
---------------------
First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install/mne_python.html).

2. Install additional dependencies
----------------------------------
You will also need to install the following additional dependencies:

- `mne-bids` to operate on BIDS data
- `coloredlogs` for nicer logging output
- `tqdm` for progress bars
- `pandas` for table creation
- `json_tricks` for handling of some analysis output
- `scikit-learn` for decoding
- `typing_extensions` if you're using a Python version older than 3.8

You can install those packages via `pip`:

??? example "Python 3.8 and newer"
    ```shell
    pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks
    ```

??? example "Older Python versions"
    ```shell
    pip install mne-bids coloredlogs tqdm pandas json_tricks scikit-learn typing_extension
    ```

3. Download the Study Template
------------------------------
TODO
