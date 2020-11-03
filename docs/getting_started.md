1. Adjust the configuration file
------------------------------
The 
Generally, there is a single `config.py` file, which contains all parameters
for the analysis of the data. Many parameters are automatically inferred from
the BIDS structure of the data. Either edit the `config.py` in-place, or create
a copy and edit the copy.

???+ info
    You should only need to touch `config.py. All other scripts should not be
    edited.


1. Inspect your dataset
-----------------------
The Study Template **only** works with BIDS-formatted raw data.

It is of great importance that

- the BIDS data are anonymized if you require anonymization,
    as the Study Template does not allow you to anonymize data.

    *This was a conscious design decision, not a technical
    limitation *per se*. If you think this decision should be
    reconsidered, please get in touch with the developers.*

- faulty channels are marked as "bad" in the BIDS dataset.
    While we *do* run automated bad channel detection in the
    Study Template, it is considered good practice to flag
    obviously problematic channels as such in the BIDS dataset.


1. Set `MNE_BIDS_STUDY_CONFIG` environment variable
---------------------------------------------------

1. Use the `Makefile` to run your analyses
-----------------------------------------


