MNE-BIDS-Pipeline processes your data in a sequential manner, i.e., one step
at a time. The next step is only run after the previous steps have been
successfully completed. There are, of course, exceptions; for example, if you
chose not to apply ICA, the respective steps will simply be omitted and we'll
directly move to the subsequent steps. The following flow chart aims to give
you a brief overview of which steps are included in the pipeline, in which
order they are run, and how we group them together.

!!! info
    All intermediate results are saved to disk for later
    inspection, and an **extensive report** is generated.

!!! info
    Analyses are conducted on individual (per-subject) as well as group level.


## :open_file_folder: Filesystem initialization and dataset inspection
```mermaid
flowchart TD
    A1[initialize the target directories] --> A2[locate empty-room recordings]
```

## :broom: Preprocessing
```mermaid
    flowchart TD
    B1[Noisy & flat channel detection] --> B2[Maxwell filter]
    B2 --> B3[Frequency filter]
    B3 --> B4[Epoch creation]
    B4 --> B5[SSP or ICA fitting]
    B5 --> B6[Artifact removal via SSP or ICA]
    B6 --> B7[Amplitude-based epoch rejection]
```

## :satellite: Sensor-space processing
```mermaid
    flowchart TD
    C1[ERP / ERF calculation] --> C2[MVPA: full epochs]
    C2 --> C3[MVPA: time-by-time decoding]
    C3 --> C4[Time-frequency decomposition]
    C4 --> C5[MVPA: CSP]
    C5 --> C6[Noise covariance estimation]
    C6 --> C7[Grand average]
```

## :brain: Source-space processing
```mermaid
    flowchart TD
    D1[BEM surface creation] --> D2[BEM solution]
    D2 --> D3[Source space creation]
    D3 --> D4[Forward model creation]
    D4 --> D5[Inverse solution]
    D5 --> D6[Grand average]
```
