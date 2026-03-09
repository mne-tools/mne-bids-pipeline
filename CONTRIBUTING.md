# Contributing to MNE-BIDS-Pipeline

Contributors to MNE-BIDS-Pipeline are expected to follow our
[Code of Conduct](https://github.com/mne-tools/.github/blob/main/CODE_OF_CONDUCT.md).

## Installation

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](http://martinos.org/mne/stable/install_mne_python.html).
Once this is done, you should be able to run this in a terminal:

`$ python -c "import mne; mne.sys_info()"`

You can then install the following additional package via `pip`. Note that
the URL points to the bleeding edge version of `mne_bids`:

`$ pip install https://github.com/mne-tools/mne-bids/zipball/main`

Now, get the pipeline through git:

`$ git clone https://github.com/mne-tools/mne-bids-pipeline.git`

If you do not know how to use git, download the pipeline as a zip file
[here](https://github.com/mne-tools/mne-bids-pipeline/archive/main.zip).

Finally, for source analysis you'll also need `FreeSurfer`, follow the
instructions on [their website](https://surfer.nmr.mgh.harvard.edu/).

Then install the packages required for testing while in the cloned repo via

```
pip install -e . --group dev
```

## Testing

### Running the tests, and continuous integration

The tests are run using `pytest`. You can run them by calling
`pytest mne_bids_pipeline` to run
all tests, or for example with `pytest -k DATASET` to run tests for a
specific dataset only, or `pytest -m "not dataset_test"` to run non-dataset
tests. To download data for a given dataset, use
`pytest -k DATASET --download`.

For every pull request or merge into the `main` branch of the
`mne-bids-pipeline`,
[CircleCI](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline)
will run tests as defined in `./circleci/config.yml`.

You can add the pre-commit hook locally after installing `lefthook` with `pip install --group dev` or similar then running `lefthook install`, or run them manually via `lefthook run pre-commit --all-files`.


### Debugging

To run the test in debugging mode, just pass `--pdb` to the `pytest` call
as usual. This will place you in debugging mode on failure.
See the
[pdb help](https://docs.python.org/3/library/pdb.html#debugger-commands)
for more commands.

### Config files

Nested in the `tests` directory is a `configs` directory, which contains
config files for specific test datasets. For example, the `config_ds001810.py`
file specifies parameters only for the `ds001810` data, which should overwrite
the more general parameters in the main `_config.py` file.
