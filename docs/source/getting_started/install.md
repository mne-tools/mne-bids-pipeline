Install MNE-Python
------------------
First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install/mne_python.html).

Install additional dependencies
-------------------------------
You will also need to install the a number of additional dependencies that are
required to run the pipeline.

??? example "Install for Python 3.8 and newer"
    Run in your terminal:
    ```shell
    pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire
    ```

??? example "Install for older Python versions"
    Run in your terminal:
    ```shell
    pip install mne-bids coloredlogs tqdm pandas json_tricks scikit-learn fire typing_extensions
    ```

??? info "Detailed list of dependencies"
    - `mne-bids` to operate on BIDS data
    - `coloredlogs` for nicer logging output
    - `tqdm` for progress bars
    - `pandas` for table creation
    - `json_tricks` for handling of some analysis output
    - `scikit-learn` for decoding
    - `fire` for the command line interface
    - `typing_extensions` if you're using a Python version older than 3.8
    - `autoreject` to automatically reject bad trials

Download MNE-BIDS-Pipeline
--------------------------
TODO
