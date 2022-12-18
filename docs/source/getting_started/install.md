
## :package: Installing MNE-BIDS-Pipeline and all dependencies

There are a few different ways to install MNE-BIDS-Pipeline, depending on how
you installed MNE-Python.

=== "MNE installer"
    :white_check_mark: Nothing to do!
    If you used the MNE-Python installer for version 1.3 or later,
    MNE-BIDS-Pipeline is already installed!

=== "conda (new environment)"
    :person_tipping_hand_tone2: We **strongly** advise you to install
    MNE-BIDS-Pipeline into a dedicated environment.

    :package: Running the following commands
    will first install `mamba`, an extremely fast drop-in replacement for `conda`, and then
    proceed to create an environment named `mne` with MNE-BIDS-Pipeline and all
    required dependencies:
    ```shell
    conda install --channel=conda-forge mamba
    mamba create --override-channels --channel=conda-forge --name=mne mne-bids-pipeline
    ```

=== "conda (existing environment)"
    :snake: If you already have a `conda` environment with MNE-Python installed following the
    [official installation instructions](https://mne.tools/stable/install/manual_install.html#installing-mne-python-with-all-dependencies),
    you can install the pipeline into the existing environment. We recommend
    using `mamba`, an extremely fast drop-in replacement for `conda`:
    ```shell
    conda install --channel=conda-forge mamba
    mamba install --override-channels --channel=conda-forge --name=mne mne-bids-pipeline
    ```

=== "pip"
    :package: Activate your Python environment and run:
    ```shell
    pip install --upgrade mne-bids-pipeline
    ```


## :mag: Testing the installation

If the installation was successful, the command-line utility `mne_bids_pipeline`
(mind the underscores!) should now be available in your Python environment.

!!! info

    `mne_bids_pipeline` will be used to operate the pipeline.

To check whether the command exists, and to verify which version of
MNE-BIDS-Pipeline is currently installed, run:

```shell
mne_bids_pipeline --version
```

**That's it! :partying_face:**

You're now ready to start using MNE-BIDS-Pipeline.
