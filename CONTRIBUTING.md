# Overview

Contributors to MNE-BIDS-Pipeline are expected to follow our
[Code of Conduct](https://github.com/mne-tools/.github/blob/main/CODE_OF_CONDUCT.md).

# Installation

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](http://martinos.org/mne/stable/install_mne_python.html).
Once this is done, you should be able to run this in a terminal:

`$ python -c "import mne; mne.sys_info()"`

You can then install the following additional packages via `pip`. Note that
the URL points to the bleeding edge version of `mne_bids`:

`$ pip install datalad`
`$ pip install https://github.com/mne-tools/mne-bids/zipball/main`

To get the test data, you need to install `git-annex` on your system. If you
installed MNE-Python via `conda`, you can simply call:

`conda install -c conda-forge git-annex`

Now, get the pipeline through git:

`$ git clone https://github.com/mne-tools/mne-bids-pipeline.git`

If you do not know how to use git, download the pipeline as a zip file
[here](https://github.com/mne-tools/mne-bids-pipeline/archive/main.zip).

Finally, for source analysis you'll also need `FreeSurfer`, follow the
instructions on [their website](https://surfer.nmr.mgh.harvard.edu/).

# Testing

## Test data

The `/tests` directory contains a module `download_test_data.py`.

If called as a script, `download_test_data` accepts a positional argument
`dataset` which can be any dataset key as specified in the module code. If no
`dataset` argument is given, the complete test data will be downloaded.

The data will then be downloaded to your path specified in the `"MNE_DATA"`
field of your
[MNE-Python config](https://mne.tools/stable/auto_tutorials/misc/plot_configuration.html#sphx-glr-auto-tutorials-misc-plot-configuration-py)
or to the `~/data` directory by defaault.

You can also call the script using `make fetch`, and you can define an
environment variable `DATASET` to specify which dataset should be downloaded

## Config files

Nested in the `/tests` directory is a `/configs` directory, which contains
config files for specific test datasets. For example, the `config_ds001810.py`
file specifies parameters only for the `ds001810` data, which should overwrite
the more general parameters in the main `_config.py` file.

## Running the tests, and continuous integration

The tests are run with help of the `tests/run_tests.py` module and the
`run_tests` function therein. You can run them by calling
`python tests/run_tests.py <arg>`, where `<arg>` can be one of the following:

- `--help`, to print the help
- `ALL`, to test all datasets
- any dataset name, similar to the description in the "test data" section above

Instead of specifying an argument via the command line, you can also define
an environment variable `DATASET` to pass your option.

For every pull request or merge into the `main` branch of the
`mne-bids-pipeline`,
[CircleCI](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline)
will run tests as defined in `./circleci/config.yml`.

## Debugging

To run the test in debugging mode, you can use the
[Python Debugger `pdb`](https://docs.python.org/3/library/pdb.html).

Simply define an environment variable `DATASET` as described in the section
above and then call:

`python -m pdb tests/run_tests.py`

This will place you in debugging mode. Type `continue` to start running the
pipelines. See the
[pdb help](https://docs.python.org/3/library/pdb.html#debugger-commands)
for more commands.
