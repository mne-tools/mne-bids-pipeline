Prepare your dataset
--------------------
The Study Template **only** works with BIDS-formatted raw data. To find out
more about BIDS and how to convert your data to the BIDS format, please see
[the documentation of MNE-BIDS](https://mne.tools/mne-bids/stable/index.html).

It is of great importance that

- the **data is anonymized** if you require anonymization,
  as the Study Template does not allow for anonymization.

    ??? info "Why?"
        This was a conscious design decision, not a technical
        limitation *per se*. If you think this decision should be
        reconsidered, please get in touch with the developers.

- **faulty channels are marked** as "bad".

    ??? info "Why?"
        While we *do* run automated bad channel detection in the
        Study Template, it is considered good practice to flag
        obviously problematic channels as such in the BIDS dataset.

Adjust your configuration file
------------------------------
The Study Template ships with a default configuration file, `config.py`.
You need to **create a copy** of that configuration file and adjust all
parameters that are relevant to your data processing and analysis.

???+ info
    You should **only** need to touch the configuration file.
    None of the scripts should be edited.

Run the Study Template
----------------------
Run the full Study Template by invoking
```shell
python run.py --steps=all --config=/path/to/your/custom_config.py
```
To only run the sensor-level, source-level, or report-generating steps, run:
```shell
python run.py --steps=sensor --config=/path/to/your/custom_config.py  # sensor-level
python run.py --steps=source --config=/path/to/your/custom_config.py  # source-level
python run.py --steps=report --config=/path/to/your/custom_config.py  # generate Reports
```
