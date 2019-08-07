# Installation

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](http://martinos.org/mne/stable/install_mne_python.html).
Once this is done, you should be able to run this in a terminal:

`$ python -c "import mne; mne.sys_info()"`

You can then install the following additional packages via `pip`. Note that
the URL points to the bleeding edge version of `mne_bids`:

`$ pip install datalad`
`$ pip install https://github.com/mne-tools/mne-bids/zipball/master`

To get the test data, you need to install `git-annex` on your system. If you
installed MNE-Python via `conda`, you can simply call:

`conda install -c conda-forge git-annex`

Now, get the study template through git:

`$ git clone https://github.com/mne-tools/mne-study-template.git`

If you do not know how to use git, download the study template as a zip file
[here](https://github.com/mne-tools/mne-study-template/archive/master.zip).

Finally, for source analysis you'll also need `FreeSurfer`, follow the
instructions on [their website](https://surfer.nmr.mgh.harvard.edu/).

## Running test data

the `/tests` directory contains a script `download_test_data.py`, which will
download all test data into the `~/data` directory on your system. You can
call the script using `make fetch`.

Nested in the `/tests` directory is a `/configs` directory, which contains
config files for specific datasets. For example, the `config_ds001810.py` file
specifies parameters only for the `ds001810` data, which should overwrite the
more general parameters in the main `config.py` file.
