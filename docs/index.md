What is the MNE Study Template?
===============================

The MNE Study Template is a full-flegded processing pipeline for your MEG and
EEG data. It operates on data stored according to the Brain Imaging Data
Structure (BIDS). The input is your raw data; the Study Template is configured
using a simple, human-readable configuration file. When run, it will conduct
preprocessing (filtering, artifact rejection), epoching, generation of evoked
responses, contrasting of experimental conditions, time-frequency analysis,
and source estimation. All intermediate results are saved to disk for later
inspection, and an extensive report is generated. Analyses are conducted on
individual (per-subject) as well as group level.

