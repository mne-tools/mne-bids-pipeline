Install MNE-Python
------------------

First, you need to make sure you have MNE-Python installed and working on your
system. See the [installation instructions](https://mne.tools/stable/install/mne_python.html).

Install the MNE-BIDS-Pipeline
-----------------------------

The following command will install the latest stable version of the
MNE-BIDS-Pipeline and all dependencies:

???+ example "Installation"
    ```shell
    pip install --upgrade mne-bids-pipeline
    ```

This also installs a command-line utility with the name `mne_bids_pipeline`
(mind the underscores!), which will be used to operate the pipeline.

To check which version of the MNE-BIDS-Pipeline is currently installed, run:

???+ example "Check the installed version"
    ```shell
    mne_bids_pipeline --version
    ```

That's it! You're now ready to start using the MNE-BIDS-Pipeline.

[Discover Basic Usage :fontawesome-solid-rocket:](basic_usage.md){: .md-button .md-button--primary }
