Install MNE-Python
------------------

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install).

Install the MNE-BIDS-Pipeline
-----------------------------

If you used the MNE-Python installer for version 1.3 or later,
MNE-BIDS-Pipeline should already be installed in the environment.

The latest stable version of the MNE-BIDS-Pipeline and all dependencies
can be installed with `pip` or `conda` the standard ways:

???+ example "Installation via pip"
    ```shell
    pip install --upgrade mne-bids-pipeline
    ```

???+ example "Installation via conda"
    ```shell
    conda install -c conda-forge mne-bids-pipeline
    ```

This installs the command-line interface `mne_bids_pipeline`
(mind the underscores!) which will be used to operate the pipeline.

To check which version of the MNE-BIDS-Pipeline is currently installed, run:

???+ example "Check the installed version"
    ```shell
    mne_bids_pipeline --version
    ```

That's it! You're now ready to start using the MNE-BIDS-Pipeline.

[Discover Basic Usage :fontawesome-solid-rocket:](basic_usage.md){: .md-button .md-button--primary }
