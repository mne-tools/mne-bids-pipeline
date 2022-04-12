Prepare your dataset
--------------------
MNE-BIDS-Pipeline **only** works with
[BIDS-formatted raw data](https://bids-specification.readthedocs.io/en/stable/). To find out
more about BIDS and how to convert your data to the BIDS format, please see
[the documentation of MNE-BIDS](https://mne.tools/mne-bids/stable/index.html).

We recommend that

- **faulty channels are marked** as "bad".

    ??? info "Why?"
        While we *do* run automated bad channel detection in the
        pipeline, it is considered good practice to flag
        obviously problematic channels as such in the BIDS dataset.

    ??? tip "How?"
        MNE-BIDS provides a convenient way to visually inspect raw data and 
        interactively mark problematic channels as bad by using the command
        ```shell
        mne-bids inspect
        ```
        Please see the MNE-BIDS documentation for more information.

- the **data is anonymized** before running the pipeline if you
  require anonymization, as the pipeline itself does not allow for anonymization.

    ??? info "Why?"
        This was a conscious design decision, not a technical
        limitation *per se*. If you think this decision should be
        reconsidered, please get in touch with the developers.

    ??? tip "How?"
        If you already have BIDS formatted data you can use
        [`mne_bids.anonymize_dataset`](https://mne.tools/mne-bids/stable/generated/mne_bids.anonymize_dataset.html#mne-bids-anonymize-dataset).
        Otherwise you can use the [`mne_bids.write_raw_bids`](https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html)
        function of MNE-BIDS that accepts an `anonymize` parameter and can be used
        to anonymize your data by removing subject-identifying information and shifting
        the measurement date by a given number of days. For example, you could use
        ```python
        from mne_bids import write_raw_bids

        write_raw_bids(..., anonymize=dict(daysback=1000))
        ```
        to shift the recording date 1000 days into the past. By default,
        information like participant handedness etc. will be removed as well.

        You can also deface your MRIs with [`mne_bids.write_anat`](https://mne.tools/mne-bids/stable/generated/mne_bids.write_anat.html):
        ```python
        from mne_bids import write_anat

        write_anat(..., landmarks=landmarks, deface=True)
        ```
        Please see [the tutorials of `mne_bids`](https://mne.tools/mne-bids/stable/use.html) for more information.

Adjust your configuration file
------------------------------

The pipeline ships with a default configuration file,
[`config.py`](https://github.com/mne-tools/mne-bids-pipeline/blob/main/config.py).
You need to **create a copy** of that configuration file and adjust all
parameters that are relevant to your data processing and analysis.

!!! warning "Do not modify the scripts and default configuration file."
    You should **only** need to modify the **copy** of the configuration file.
    **None of the scripts from the pipeline should be edited!**

Run the pipeline
----------------

???+ example "Run the full pipeline"
    To run the full pipeline, execute the following command in your
    terminal:
    ```shell
    python run.py --config=/path/to/your/custom_config.py
    ```

??? example "Run only parts of the pipeline"
    Run only the preprocessing steps:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=preprocessing
    ```

    Run only the sensor-level processing steps:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=sensor
    ```

    Run only the source-level (inverse solution) processing steps:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=source
    ```

    Only generate the report:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=report
    ```

    (Re-)run ICA:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=preprocessing/ica
    ```

    You can also run multiple steps with one command by separating different
    steps by a comma. For example, to run preprocessing and sensor-level
    processing steps using a single command, do:
    ```shell
    python run.py --config=/path/to/your/custom_config.py --steps=preprocessing,sensor
    ```

You can directly visit our [examples page](../examples/examples.html) to see some configuration files
and the corresponding results.
