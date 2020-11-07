What is the MNE Study Template?
===============================

**The MNE Study Template is a full-flegded processing pipeline for your MEG and
EEG data.**

It operates on raw data stored according to the Brain Imaging Data
Structure (BIDS). Processing is controlled using a simple human-readable
configuration file.

??? list "List of features"
    
    - preprocessing (filtering, artifact rejection)
    - epoching
    - generation of evoked responses
    - contrasting of experimental conditions
    - time-frequency analysis
    - source estimation

    All intermediate results are saved to disk for later
    inspection, and an **extensive report** is generated.

    Analyses are conducted on individual (per-subject) as well as group level.

<a  href="getting_started/install.html"
    title="Installation"
    class="md-button md-button--primary">👉 Let's get started</a>


*[MEG]: magnetoencaphalography
*[EEG]: electroencephalography
