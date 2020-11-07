Install MNE-Python
------------------
First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install/mne_python.html).

Install additional dependencies
-------------------------------
You will also need to install the a number of additional dependencies that are
required to run the Study Template.

=== "Install for Python 3.8 and newer"
    ???+ example "Run in terminal"
        ```shell
        pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire
        ```

=== "Install for older Python versions"

    ???+ example "Run in terminal"
        ```shell
        pip install mne-bids coloredlogs tqdm pandas json_tricks scikit-learn fire typing_extension
        ```

=== "Detailed list of dependencies"

    - `mne-bids` to operate on BIDS data
    - `coloredlogs` for nicer logging output
    - `tqdm` for progress bars
    - `pandas` for table creation
    - `json_tricks` for handling of some analysis output
    - `scikit-learn` for decoding
    - `fire` for the command line interface
    - `typing_extensions` if you're using a Python version older than 3.8


Download the Study Template
---------------------------
TODO
