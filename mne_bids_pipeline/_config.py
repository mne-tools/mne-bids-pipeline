# Default settings for data processing and analysis.

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

import pandas as pd
from annotated_types import Ge, Interval, Len, MinLen
from mne import Covariance
from mne_bids import BIDSPath

from mne_bids_pipeline.typing import (
    ArbitraryContrast,
    DigMontageType,
    FloatArrayLike,
    PathLike,
    UniqueSequence,
)

# %%
# # General settings

bids_root: PathLike | None = None
"""
Specify the BIDS root directory. Pass an empty string or ```None` to use
the value specified in the `BIDS_ROOT` environment variable instead.
Raises an exception if the BIDS root has not been specified.

???+ example "Example"
    ``` python
    bids_root = '/path/to/your/bids_root'  # Use this to specify a path here.
    bids_root = None  # Make use of the `BIDS_ROOT` environment variable.
    ```
"""

deriv_root: PathLike | None = None
"""
The root of the derivatives directory in which the pipeline will store
the processing results. If `None`, this will be
`derivatives/mne-bids-pipeline` inside the BIDS root.

!!! info
    If specified and you wish to run the source analysis steps, you must
    set [`subjects_dir`][mne_bids_pipeline._config.subjects_dir] as well.
"""

subjects_dir: PathLike | None = None
"""
Path to the directory that contains the FreeSurfer reconstructions of all
subjects. Specifically, this defines the `SUBJECTS_DIR` that is used by
FreeSurfer.

- When running the `freesurfer` processing step to create the
  reconstructions from anatomical scans in the BIDS dataset, the
  output will be stored in this directory.
- When running the source analysis steps, we will look for the surfaces in this
  directory and also store the BEM surfaces there.

If `None`, this will default to
[`bids_root`][mne_bids_pipeline._config.bids_root]`/derivatives/freesurfer/subjects`.

!!! info
    This setting is required if you specify
    [`deriv_root`][mne_bids_pipeline._config.deriv_root]
    and want to run the source analysis steps.
"""

interactive: bool = False
"""
If True, the steps will provide some interactive elements, such as
figures. If running the steps from a notebook or Spyder,
run `%matplotlib qt` in the command line to open the figures in a separate
window.

!!! info
    Enabling interactive mode deactivates parallel processing.
"""

sessions: list[str] | Literal["all"] = "all"
"""
The sessions to process. If `'all'`, will process all sessions found in the
BIDS dataset.
"""

allow_missing_sessions: bool = False
"""
Whether to continue processing the dataset if some combinations of `subjects` and
`sessions` are missing.
"""

task: str = ""
"""
The task to process.
"""

task_is_rest: bool = False
"""
Whether the task should be treated as resting-state data.
"""

runs: Sequence[str] | Literal["all"] = "all"
"""
The runs to process. If `'all'`, will process all runs found in the
BIDS dataset.
"""

exclude_runs: dict[str, list[str]] | None = None
"""
Specify runs to exclude from analysis, for each participant individually.

???+ example "Example"
    ```python
    exclude_runs = None  # Include all runs.
    exclude_runs = {'01': ['02']}  # Exclude run 02 of subject 01.
    ```

???+ info "Good Practice / Advice"
    Keep track of the criteria leading you to exclude
    a run (e.g. too many movements, missing blocks, aborted experiment,
    did not understand the instructions, etc.).
"""

crop_runs: tuple[float, float] | None = None
"""
Crop the raw data of each run to the specified time interval `[tmin, tmax]`,
in seconds. The runs will be cropped before Maxwell or frequency filtering is
applied. If `None`, do not crop the data.
"""

acq: str | None = None
"""
The BIDS `acquisition` entity.
"""

proc: str | None = None
"""
The BIDS `processing` entity.
"""

rec: str | None = None
"""
The BIDS `recording` entity.
"""

space: str | None = None
"""
The BIDS `space` entity.
"""

subjects: Sequence[str] | Literal["all"] = "all"
"""
Subjects to analyze. If `'all'`, include all subjects. To only
include a subset of subjects, pass a list of their identifiers. Even
if you plan on analyzing only a single subject, pass their identifier
as a list.

Please note that if you intend to EXCLUDE only a few subjects, you
should consider setting `subjects = 'all'` and adding the
identifiers of the excluded subjects to `exclude_subjects` (see next
section).

???+ example "Example"
    ```python
    subjects = 'all'  # Include all subjects.
    subjects = ['05']  # Only include subject 05.
    subjects = ['01', '02']  # Only include subjects 01 and 02.
    ```
"""

exclude_subjects: Sequence[str] = []
"""
Specify subjects to exclude from analysis. The MEG empty-room mock-subject
is automatically excluded from regular analysis.

???+ info "Good Practice / Advice"
    Keep track of the criteria leading you to exclude
    a participant (e.g. too many movements, missing blocks, aborted experiment,
    did not understand the instructions, etc, ...)
    The `emptyroom` subject will be excluded automatically.
"""

process_empty_room: bool = True
"""
Whether to apply the same pre-processing steps to the empty-room data as
to the experimental data (up until including frequency filtering). This
is required if you wish to use the empty-room recording to estimate noise
covariance (via `noise_cov='emptyroom'`). The empty-room recording
corresponding to the processed experimental data will be retrieved
automatically.
"""

process_rest: bool = True
"""
Whether to apply the same pre-processing steps to the resting-state data as
to the experimental data (up until including frequency filtering). This
is required if you wish to use the resting-state recording to estimate noise
covariance (via `noise_cov='rest'`).
"""

ch_types: Annotated[Sequence[Literal["meg", "mag", "grad", "eeg"]], Len(1, 4)] = []
"""
The channel types to consider.

???+ example "Example"
    ```python
    # Use EEG channels:
    ch_types = ['eeg']

    # Use magnetometer and gradiometer MEG channels:
    ch_types = ['mag', 'grad']

    # Use MEG and EEG channels:
    ch_types = ['meg', 'eeg']
    ```
"""

data_type: Literal["meg", "eeg"] | None = None
"""
The BIDS data type.

For MEG recordings, this will usually be 'meg'; and for EEG, 'eeg'.
However, if your dataset contains simultaneous recordings of MEG and EEG,
stored in a single file, you will typically need to set this to 'meg'.
If `None`, we will assume that the data type matches the channel type.

???+ example "Example"
    The dataset contains simultaneous recordings of MEG and EEG, and we only
    wish to process the EEG data, which is stored inside the MEG files:

    ```python
    ch_types = ['eeg']
    data_type = 'meg'
    ```

    The dataset contains simultaneous recordings of MEG and EEG, and we only
    wish to process the gradiometer data:

    ```python
    ch_types = ['grad']
    data_type = 'meg'  # or data_type = None
    ```

    The dataset contains only EEG data:

    ```python
    ch_types = ['eeg']
    data_type = 'eeg'  # or data_type = None
    ```
"""

eog_channels: Sequence[str] | None | dict[str, Sequence[str] | None] = None
"""
Specify EOG channels to use, or create virtual EOG channels.

Allows the specification of custom channel names that shall be used as
(virtual) EOG channels. For example, say you recorded EEG **without** dedicated
EOG electrodes, but with some EEG electrodes placed close to the eyes, e.g.
Fp1 and Fp2. These channels can be expected to have captured large quantities
of ocular activity, and you might want to use them as "virtual" EOG channels,
while also including them in the EEG analysis. By default, MNE won't know that
these channels are suitable for recovering EOG, and hence won't be able to
perform tasks like automated blink removal, unless a "true" EOG sensor is
present in the data as well. Specifying channel names here allows MNE to find
the respective EOG signals based on these channels.

You can specify one or multiple channel names. Each will be treated as if it
were a dedicated EOG channel, without excluding it from any other analyses.

If `None`, only actual EOG channels will be used for EOG recovery.

If there are multiple actual EOG channels in your data, and you only specify
a subset of them here, only this subset will be used during processing.

A dictionary can be provided to specify subject and/or session-level EOG,
with subjects (and optionally session) as keys and a sequence of channels as
values (see Examples). Use "default" as a key to set channels for all non-
specified subjects/sessions

???+ example "Example"
    Treat `Fp1` as virtual EOG channel:
    ```python
    eog_channels = ['Fp1']
    ```

    Treat `Fp1` and `Fp2` as virtual EOG channels:
    ```python
    eog_channels = ['Fp1', 'Fp2']
    ```

    Per default use `LEOG`, but for sub-04 use Fp1 and for sub-05 ignore EOG:
    ```python
    eog_channels = dict()
    eog_channels["default"] = ['LEOG']
    eog_channels['sub-04'] = ['Fp1']
    eog_channels['sub-05'] = []
    ```
    Note that `collections.defaultdict` cannot be used because it causes problems
    with pickling, which is used under the hood for caching and parallelization.
"""

eeg_bipolar_channels: dict[str, tuple[str, str]] | None = None
"""
Combine two channels into a bipolar channel, whose signal is the **difference**
between the two combined channels, and add it to the data.
A typical use case is the combination of two EOG channels – for example, a
left and a right horizontal EOG – into a single, bipolar EOG channel. You need
to pass a dictionary whose **keys** are the name of the new bipolar channel you
wish to create, and whose **values** are tuples consisting of two strings: the
name of the channel acting as anode and the name of the channel acting as
cathode, i.e. `{'ch_name': ('anode', 'cathode')}`. You can request
to construct more than one bipolar channel by specifying multiple key/value
pairs. See the examples below.

Can also be `None` if you do not want to create bipolar channels.

!!! info
    The channels used to create the bipolar channels are **not** automatically
    dropped from the data. To drop channels, set `drop_channels`.

???+ example "Example"
    Combine the existing channels `HEOG_left` and `HEOG_right` into a new,
    bipolar channel, `HEOG`:
    ```python
    eeg_add_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right')}
    ```

    Create two bipolar channels, `HEOG` and `VEOG`:
    ```python
    eeg_add_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
                                'VEOG': ('VEOG_lower', 'VEOG_upper')}
    ```
"""

eeg_reference: Literal["average"] | str | Sequence[str] = "average"
"""
The EEG reference to use. If `average`, will use the average reference,
i.e. the average across all channels. If a string, must be the name of a single
channel. To use multiple channels as reference, set to a list of channel names.

???+ example "Example"
    Use the average reference:
    ```python
    eeg_reference = 'average'
    ```

    Use the `P9` channel as reference:
    ```python
    eeg_reference = 'P9'
    ```

    Use the average of the `P9` and `P10` channels as reference:
    ```python
    eeg_reference = ['P9', 'P10']
    ```
"""

eeg_template_montage: str | DigMontageType | None = None
"""
In situations where you wish to process EEG data and no individual
digitization points (measured channel locations) are available, you can apply
a "template" montage. This means we will assume the EEG cap was placed
either according to an international system like 10/20, or as suggested by
the cap manufacturers in their respective manual.

Please be aware that the actual cap placement most likely deviated somewhat
from the template, and, therefore, source reconstruction may be impaired.

If `None`, do not apply a template montage. If a string, must be the
name of a built-in template montage in MNE-Python.
You can find an overview of supported template montages at
https://mne.tools/stable/generated/mne.channels.make_standard_montage.html

!!! warning
    If the data contains channel names that are not part of the template montage, the
    pipeline run will fail with an error message. You must either pick a different
    montage or remove those channels via
    [`drop_channels`][mne_bids_pipeline._config.drop_channels] to continue.


???+ example "Example"
    Do not apply template montage:
    ```python
    eeg_template_montage = None
    ```

    Apply 64-channel Biosemi 10/20 template montage:
    ```python
    eeg_template_montage = 'biosemi64'
    ```
"""

drop_channels: Sequence[str] = []
"""
Names of channels to remove from the data. This can be useful, for example,
if you have added a new bipolar channel via `eeg_bipolar_channels` and now wish
to remove the anode, cathode, or both; or if your selected EEG template montage
doesn't contain coordinates for some channels.

???+ example "Example"
    Exclude channels `Fp1` and `Cz` from processing:
    ```python
    drop_channels = ['Fp1', 'Cz]
    ```
"""

analyze_channels: Literal["all", "ch_types"] | Annotated[Sequence[str], MinLen(1)] = (
    "ch_types"
)
"""
The names of the channels to analyze during ERP/ERF and time-frequency analysis
steps. For certain paradigms, e.g. EEG ERP research, it is common to constrain
sensor-space analysis to only a few specific sensors. If `'all'`, do not
exclude any channels (except for those selected for removal via the
`drop_channels` setting; use with caution as this can include things like STIM
channels during the decoding step). If 'ch_types' (default), restrict to the
channels listed in the `ch_types` parameter. The constraint will be applied to
all sensor-level analyses after the preprocessing stage, but not to the
preprocessing stage itself, nor to the source analysis stage.

???+ example "Example"
    Only use channel `Pz` for ERP, evoked contrasts, time-by-time
    decoding, and time-frequency analysis:
    ```python
    analyze_channels = ['Pz']
    ```
"""

reader_extra_params: dict[str, Any] = {}
"""
Parameters to be passed to `read_raw_bids()` calls when importing raw data.

???+ example "Example"
    Enforce units for EDF files:
    ```python
    reader_extra_params = {"units": "uV"}
    ```
"""

read_raw_bids_verbose: Literal["error"] | None = None
"""
Verbosity level to pass to `read_raw_bids(..., verbose=read_raw_bids_verbose)`.
If you know your dataset will contain files that are not perfectly BIDS
compliant (e.g., "Did not find any meg.json..."), you can set this to
`'error'` to suppress warnings emitted by read_raw_bids.
"""

plot_psd_for_runs: Literal["all"] | Sequence[str] = "all"
"""
For which runs to add a power spectral density (PSD) plot to the generated
report. This can take a considerable amount of time if you have many long
runs. In this case, specify the runs, or pass an empty list to disable raw PSD
plotting.
"""

random_state: int | None = 42
"""
You can specify the seed of the random number generator (RNG).
This setting is passed to the ICA algorithm and to the decoding function,
ensuring reproducible results. Set to `None` to avoid setting the RNG
to a defined state.
"""

# %%
# # Preprocessing

# ## Break detection

find_breaks: bool = False
"""
During an experimental run, the recording might be interrupted by breaks of
various durations, e.g. to allow the participant to stretch, blink, and swallow
freely. During these periods, large-scale artifacts are often picked up by the
recording system. These artifacts can impair certain stages of processing, e.g.
the peak-detection algorithms we use to find EOG and ECG activity. In some
cases, even the bad channel detection algorithms might not function optimally.
It is therefore advisable to mark such break periods for exclusion at early
processing stages.

If `True`, try to mark breaks by finding segments of the data where no
experimental events have occurred. This will then add annotations with the
description `BAD_break` to the continuous data, causing these segments to be
ignored in all following processing steps.

???+ example "Example"
    Automatically find break periods, and annotate them as `BAD_break`.
    ```python
    find_breaks = True
    ```

    Disable break detection.
    ```python
    find_breaks = False
    ```
"""

min_break_duration: float = 15.0
"""
The minimal duration (in seconds) of a data segment without any experimental
events for it to be considered a "break". Note that the minimal duration of the
generated `BAD_break` annotation will typically be smaller than this, as by
default, the annotation will not extend across the entire break.
See [`t_break_annot_start_after_previous_event`][mne_bids_pipeline._config.t_break_annot_start_after_previous_event]
and [`t_break_annot_stop_before_next_event`][mne_bids_pipeline._config.t_break_annot_stop_before_next_event]
to control this behavior.

???+ example "Example"
    Periods between two consecutive experimental events must span at least
    `15` seconds for this period to be considered a "break".
    ```python
    min_break_duration = 15.
    ```
"""  # noqa : E501

t_break_annot_start_after_previous_event: float = 5.0
"""
Once a break of at least
[`min_break_duration`][mne_bids_pipeline._config.min_break_duration]
seconds has been discovered, we generate a `BAD_break` annotation that does not
necessarily span the entire break period. Instead, you will typically want to
start it some time after the last event before the break period, as to not
unnecessarily discard brain activity immediately following that event.

This parameter controls how much time (in seconds) should pass after the last
pre-break event before we start annotating the following segment of the break
period as bad.

???+ example "Example"
    Once a break period has been detected, add a `BAD_break` annotation to it,
    starting `5` seconds after the latest pre-break event.
    ```python
    t_break_annot_start_after_previous_event = 5.
    ```

    Start the `BAD_break` annotation immediately after the last pre-break
    event.
    ```python
    t_break_annot_start_after_previous_event = 0.
    ```
"""

t_break_annot_stop_before_next_event: float = 5.0
"""
Similarly to how
[`t_break_annot_start_after_previous_event`][mne_bids_pipeline._config.t_break_annot_start_after_previous_event]
controls the "gap" between beginning of the break period and `BAD_break`
annotation onset,  this parameter controls how far the annotation should extend
toward the first experimental event immediately following the break period
(in seconds). This can help not to waste a post-break trial by marking its
pre-stimulus period as bad.

???+ example "Example"
    Once a break period has been detected, add a `BAD_break` annotation to it,
    starting `5` seconds after the latest pre-break event.
    ```python
    t_break_annot_start_after_previous_event = 5.
    ```

    Start the `BAD_break` annotation immediately after the last pre-break
    event.
    ```python
    t_break_annot_start_after_previous_event = 0.
    ```
"""

# %%
# ## Bad channel detection
#
# !!! warning
#     This functionality will soon be removed from the pipeline, and
#     will be integrated into MNE-BIDS.
#
# "Bad", i.e. flat and overly noisy channels, can be automatically detected
# using a procedure inspired by the commercial MaxFilter by Elekta. First,
# a copy of the data is low-pass filtered at 40 Hz. Then, channels with
# unusually low variability are flagged as "flat", while channels with
# excessively high variability are flagged as "noisy". Flat and noisy channels
# are marked as "bad" and excluded from subsequent analysis. See
# :func:`mne.preprocssessing.find_bad_channels_maxwell` for more information
# on this procedure. The list of bad channels detected through this procedure
# will be merged with the list of bad channels already present in the dataset,
# if any.

find_flat_channels_meg: bool = False
"""
Auto-detect "flat" channels (i.e. those with unusually low variability) and
mark them as bad.
"""

find_noisy_channels_meg: bool = False
"""
Auto-detect "noisy" channels and mark them as bad.
"""


find_bad_channels_extra_kws: dict[str, Any] = {}

"""
A dictionary of extra kwargs to pass to `mne.preprocessing.find_bad_channels_maxwell`
. If kwargs are passed here that have dedicated config settings already, an error
will be raised.
For full documentation of the bad channel detection:
https://mne.tools/stable/generated/mne.preprocessing.find_bad_channels_maxwell
"""


# %%
# ## Maxwell filter

use_maxwell_filter: bool = False
"""
Whether or not to use [Maxwell filtering][mne.preprocessing.maxwell_filter] to
preprocess the data.

!!! warning
    If the data were recorded with internal active compensation (MaxShield),
    they need to be run through Maxwell filter to avoid distortions.
    Bad channels need to be set through BIDS channels.tsv and / or via the
    `find_flat_channels_meg` and `find_noisy_channels_meg` options above
    before applying Maxwell filter.
"""

mf_st_duration: float | None = None
"""
There are two kinds of Maxwell filtering: SSS (signal space separation) and
tSSS (temporal signal space separation)
(see [Taulu et al., 2004](http://cds.cern.ch/record/709081/files/0401166.pdf)).

If not None, apply spatiotemporal SSS (tSSS) with specified buffer
duration (in seconds). MaxFilter™'s default is 10.0 seconds in v2.2.
Spatiotemporal SSS acts as implicitly as a high-pass filter where the
cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
buffers are generally better as long as your system can handle the
higher memory usage. To ensure that each window is processed
identically, choose a buffer length that divides evenly into your data.
Any data at the trailing edge that doesn't fit evenly into a whole
buffer window will be lumped into the previous buffer.

???+ info "Good Practice / Advice"
    If you are interested in low frequency activity (<0.1Hz), avoid using
    tSSS and set `mf_st_duration` to `None`.

    If you are interested in low frequency above 0.1 Hz, you can use the
    default `mf_st_duration` to 10 s, meaning it acts like a 0.1 Hz
    high-pass filter.

???+ example "Example"
    ```python
    mf_st_duration = None
    mf_st_duration = 10.  # to apply tSSS with 0.1Hz highpass filter.
    ```
"""

mf_st_correlation: float = 0.98
"""
The correlation limit for spatio-temporal SSS (tSSS).

???+ example "Example"
    ```python
    st_correlation = 0.98
    ```
"""

mf_head_origin: Literal["auto"] | FloatArrayLike = "auto"
"""
`mf_head_origin` : array-like, shape (3,) | 'auto'
Origin of internal and external multipolar moment space in meters.
If 'auto', it will be estimated from headshape points.
If automatic fitting fails (e.g., due to having too few digitization
points), consider separately calling the fitting function with different
options or specifying the origin manually.

???+ example "Example"
    ```python
    mf_head_origin = 'auto'
    ```
"""

mf_destination: Literal["reference_run", "twa"] | FloatArrayLike = "reference_run"
"""
Despite all possible care to avoid movements in the MEG, the participant
will likely slowly drift down from the Dewar or slightly shift the head
around in the course of the recording session. Hence, to take this into
account, we are realigning all data to a single position. For this, you can:

1. Choose a reference run. Often one from the middle of the recording session
   is a good choice. Set `mf_destination = "reference_run" and then set
   [`config.mf_reference_run`][mne_bids_pipeline._config.mf_reference_run].
   This will result in a device-to-head transformation that differs between
   subjects.
2. Choose a standard position in the MEG coordinate frame. For this, pass
   a 4x4 transformation matrix for the device-to-head
   transform. This will result in a device-to-head transformation that is
   the same across all subjects.

    ???+ example "A Standardized Position"
        ```python
        from mne.transforms import translation
        mf_destination = translation(z=0.04)
        ```
3. Compute the time-weighted average head position across all runs in a session,
   and use that as the destination coordinates for each run. This will result in a
   device-to-head transformation that differs between sessions within each subject.
"""

mf_int_order: int = 8
"""
Internal order for the Maxwell basis. Can increase or decrease for datasets where
neural signals with higher or lower spatial complexity are expected.
Per MNE, the default values are appropriate for most use cases.
"""

mf_ext_order: int = 3
"""
External order for the Maxwell basis. Can increase or decrease for datasets where
environmental artifacts with higher or lower spatial complexity are expected.
Per MNE, the default values are appropriate for most use cases.
"""

mf_reference_run: str | None = None
"""
Which run to take as the reference for adjusting the head position of all
runs when [`mf_destination="reference_run"`][mne_bids_pipeline._config.mf_destination].
If `None`, pick the first run.

???+ example "Example"
    ```python
    mf_reference_run = '01'  # Use run "01"
    ```
"""

mf_cal_fname: str | None = None
"""
!!! warning
     This parameter should only be used for BIDS datasets that don't store
     the fine-calibration file
     [according to BIDS](https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#cross-talk-and-fine-calibration-files).
Path to the Maxwell Filter calibration file. If `None`, the recommended
location is used.
???+ example "Example"
    ```python
    mf_cal_fname = '/path/to/your/file/calibration_cal.dat'
    ```
"""

mf_cal_missing: Literal["ignore", "warn", "raise"] = "raise"
"""
How to handle the situation where the MEG device's fine calibration file is missing.
Possible options are to ignore the missing file (as may be appropriate for OPM data),
issue a warning, or raise an error.
"""

mf_ctc_fname: str | None = None
"""
Path to the Maxwell Filter cross-talk file. If `None`, the recommended
location is used.

!!! warning
     This parameter should only be used for BIDS datasets that don't store
     the cross-talk file
     [according to BIDS](https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#cross-talk-and-fine-calibration-files).
???+ example "Example"
    ```python
    mf_ctc_fname = '/path/to/your/file/crosstalk_ct.fif'
    ```
"""

mf_ctc_missing: Literal["ignore", "warn", "raise"] = "raise"
"""
How to handle the situation where the MEG device's cross-talk file is missing. Possible
options are to ignore the missing file (as may be appropriate for OPM data), issue a
warning, or raise an error (appropriate for data from Electa/Neuromag/MEGIN systems).
"""

mf_esss: int = 0
"""
Number of extended SSS (eSSS) basis projectors to use from empty-room data.
"""

mf_esss_reject: dict[str, float] | None = None
"""
Rejection parameters to use when computing the extended SSS (eSSS) basis.
"""

mf_mc: bool = False
"""
If True, perform movement compensation on the data.
"""

mf_mc_t_step_min: float = 0.01
"""
Minimum time step to use during cHPI coil amplitude estimation.
"""

mf_mc_t_window: float | Literal["auto"] = "auto"
"""
The window to use during cHPI coil amplitude estimation and in cHPI filtering.
Can be "auto" to autodetect a reasonable value or a float (in seconds).
"""

mf_mc_gof_limit: float = 0.98
"""
Minimum goodness of fit to accept for each cHPI coil.
"""

mf_mc_dist_limit: float = 0.005
"""
Minimum distance (m) to accept for cHPI position fitting.
"""

mf_mc_rotation_velocity_limit: float | None = None
"""
The rotation velocity limit (degrees/second) to use when annotating
movement-compensated data. If `None`, no annotations will be added.
"""

mf_mc_translation_velocity_limit: float | None = None
"""
The translation velocity limit (meters/second) to use when annotating
movement-compensated data. If `None`, no annotations will be added.
"""

mf_filter_chpi: bool | None = None
"""
Use mne.chpi.filter_chpi after Maxwell filtering. Can be None to use
the same value as [`mf_mc`][mne_bids_pipeline._config.mf_mc].
Only used when [`use_maxwell_filter=True`][mne_bids_pipeline._config.use_maxwell_filter]
"""

mf_extra_kws: dict[str, Any] = {}
"""
A dictionary of extra kwargs to pass to `mne.preprocessing.maxwell_filter`. If kwargs
are passed here that have dedicated config settings already, an error will be raised.
For full documentation of the Maxwell filter:
https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter
"""

# ## Filtering & resampling

# ### Filtering
#
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border (or edge) effects.
#
# If you use this pipeline for evoked responses, you could consider
# a low-pass filter cut-off of h_freq = 40 Hz
# and possibly a high-pass filter cut-off of l_freq = 1 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band.
# Note that highpass filtering is not necessarily recommended as it can
# distort waveforms of evoked components, or simply wash out any low
# frequency that can may contain brain signal. It can also act as
# a replacement for baseline correction in Epochs. See below.
#
# If you use this pipeline for time-frequency analysis, a default filtering
# could be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! 😇

l_freq: float | None = None
"""
The low-frequency cut-off in the highpass filtering step.
Keep it `None` if no highpass filtering should be applied.
"""

h_freq: float | None = 40.0
"""
The high-frequency cut-off in the lowpass filtering step.
Keep it `None` if no lowpass filtering should be applied.
"""

l_trans_bandwidth: float | Literal["auto"] = "auto"
"""
Specifies the transition bandwidth of the
highpass filter. By default it's `'auto'` and uses default MNE
parameters.
"""

h_trans_bandwidth: float | Literal["auto"] = "auto"
"""
Specifies the transition bandwidth of the
lowpass filter. By default it's `'auto'` and uses default MNE
parameters.
"""

notch_freq: float | Sequence[float] | None = None
"""
Notch filter frequency. More than one frequency can be supplied, e.g. to remove
harmonics. Keep it `None` if no notch filter should be applied.

!!! info
    The notch filter will be applied before high- and lowpass filtering.

???+ example "Example"
    Remove line noise at 50 Hz:
    ```python
    notch_freq = 50
    ```
    Remove line noise at 50 Hz and its (sub-)harmonics
    ```python
    notch_freq = [25, 50, 100, 150]
    ```
"""

notch_trans_bandwidth: float = 1.0
"""
Specifies the transition bandwidth of the notch filter. The default is `1.`.
"""

notch_widths: float | Sequence[float] | None = None
"""
Specifies the width of each stop band. `None` uses the MNE default.
"""

zapline_fline: float | None = None
"""
Specifies frequency to remove using Zapline filtering. If None, zapline will not
be used.
"""

zapline_iter: bool = False
"""
Specifies if the iterative version of the Zapline algorithm should be run.
"""

notch_extra_kws: dict[str, Any] = {}
"""
A dictionary of extra kwargs to pass to `mne.filter.notch_filter`. If kwargs
are passed here that have dedicated config settings already, an error will be raised.
For full documentation of the notch filter:
https://mne.tools/stable/generated/mne.filter.notch_filter.
"""

bandpass_extra_kws: dict[str, Any] = {}
"""
A dictionary of extra kwargs to pass to `mne.filter.filter_data`. If kwargs
are passed here that have dedicated config settings already, an error will be raised.
For full documatation of the bandpass filter:
https://mne.tools/stable/generated/mne.filter.filter_data
"""

# ### Resampling
#
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data.

raw_resample_sfreq: float | None = None
"""
Specifies at which sampling frequency the data should be resampled.
If `None`, then no resampling will be done.

???+ example "Example"
    ```python
    raw_resample_sfreq = None  # no resampling
    raw_resample_sfreq = 500  # resample to 500Hz
    ```
"""

epochs_decim: int = 1
"""
Says how much to decimate data at the epochs level.
It is typically an alternative to the `resample_sfreq` parameter that
can be used for resampling raw data. `1` means no decimation.

???+ info "Good Practice / Advice"
    Decimation requires to lowpass filtered the data to avoid aliasing.
    Note that using decimation is much faster than resampling.

???+ example "Example"
    ```python
    epochs_decim = 1  # no decimation
    epochs_decim = 4  # decimate by 4, i.e., divide sampling frequency by 4
    ```
"""


# ## Epoching

rename_events: dict[str, str] = dict()
"""
A dictionary specifying which events in the BIDS dataset to rename upon
loading, and before processing begins.

Pass an empty dictionary to not perform any renaming.

???+ example "Example"
    Rename `audio_left` in the BIDS dataset to `audio/left` in the
    pipeline:
    ```python
    rename_events = {'audio_left': 'audio/left'}
    ```
"""

on_rename_missing_events: Literal["ignore", "warn", "raise"] = "raise"
"""
How to handle the situation where you specified an event to be renamed via
`rename_events`, but this particular event is not present in the data. By
default, we will raise an exception to avoid accidental mistakes due to typos;
however, if you're sure what you're doing, you may change this to `'warn'`
to only get a warning instead, or `'ignore'` to ignore it completely.
"""

event_repeated: Literal["error", "drop", "merge"] = "error"
"""
How to handle repeated events. We call events "repeated" if more than one event
occurred at the exact same time point. Currently, MNE-Python cannot handle
this situation gracefully when trying to create epochs, and will throw an
error. To only keep the event of that time point ("first" here referring to
the order that events appear in `*_events.tsv`), pass `'drop'`. You can also
request to create a new type of event by merging repeated events by setting
this to `'merge'`.

!!! warning
    The `'merge'` option is entirely untested in the MNE BIDS Pipeline as of
    April 1st, 2021.
"""

epochs_custom_metadata: pd.DataFrame | dict[str, Any] | None = None

"""
Pandas `DataFrame` containing custom metadata. The custom metadata will be
horizontally joined with the metadata generated from `events.tsv`.
The number of rows in the custom metadata must match the number of rows in
the events metadata (after filtering by `conditions`).

The metadata can also be formatted as a `dict`, with keys being the `subject`,
`session`, and/or `task`, and the values being a `DataFrame`. e.g.:
```python
epochs_custom_metadata = {'sub-01': {'ses-01': {'task-taskA': my_DataFrame}}}
epochs_custom_metadata = {'ses-01': my_DataFrame1, 'ses-02': my_DataFrame2}
```

If None, don't use custom metadata.
"""


epochs_metadata_tmin: float | str | list[str] | None = None
"""
The beginning of the time window used for epochs metadata generation. This setting
controls the `tmin` value passed to
[`mne.epochs.make_metadata`](https://mne.tools/stable/generated/mne.epochs.make_metadata.html).

If a float, the time in seconds relative to the time-locked event of the respective
epoch. Negative indicate times before, positive values indicate times after the
time-locked event.

If a string or a list of strings, the name(s) of events marking the start of time
window.

If `None`, use the first time point of the epoch.

???+ info
     Note that `None` here behaves differently than `tmin=None` in
     `mne.epochs.make_metadata`. To achieve the same behavior, pass the name(s) of the
     time-locked events instead.

"""

epochs_metadata_tmax: float | str | list[str] | None = None
"""
Same as `epochs_metadata_tmin`, but specifying the **end** of the time
window for metadata generation.
"""

epochs_metadata_keep_first: Sequence[str] | None = None
"""
Event groupings using hierarchical event descriptors (HEDs) for which to store
the time of the **first** occurrence of any event of this group in a new column
with the group name, and the **type** of that event in a column named after the
group, but with a `first_` prefix. If `None` (default), no event
aggregation will take place and no new columns will be created.

???+ example "Example"
    Assume you have two response events types, `response/left` and
    `response/right`; in some trials, both responses occur, because the
    participant pressed both buttons. Now, you want to keep the first response
    only. To achieve this, set
    ```python
    epochs_metadata_keep_first = ['response']
    ```
    This will add two new columns to the metadata: `response`, indicating
    the **time** relative to the time-locked event; and `first_response`,
    depicting the **type** of event (`'left'` or `'right'`).

    You may also specify a grouping for multiple event types:
    ```python
    epochs_metadata_keep_first = ['response', 'stimulus']
    ```
    This will add the columns `response`, `first_response`, `stimulus`,
    and `first_stimulus`.
"""

epochs_metadata_keep_last: Sequence[str] | None = None
"""
Same as `epochs_metadata_keep_first`, but for keeping the **last**
occurrence of matching event types. The columns indicating the event types
will be named with a `last_` instead of a `first_` prefix.
"""

epochs_metadata_query: str | None = None
"""
A [metadata query](https://mne.tools/stable/auto_tutorials/epochs/30_epochs_metadata.html)
specifying which epochs to keep. If the query fails because it refers to an
unknown metadata column, a warning will be emitted and all epochs will be kept.

???+ example "Example"
    Only keep epochs without a `response_missing` event:
    ```python
    epochs_metadata_query = ['response_missing.isna()']
    ```
"""

conditions: Sequence[str] | dict[str, str] | None = None
"""
The time-locked events based on which to create evoked responses.
This can either be name of the experimental condition as specified in the
BIDS `*_events.tsv` file; or the name of condition *groups*, if the condition
names contain the (MNE-specific) group separator, `/`. See the [Subselecting
epochs tutorial](https://mne.tools/stable/auto_tutorials/epochs/plot_10_epochs_overview.html#subselecting-epochs)
for more information.

Passing a dictionary allows to assign a name to map a complex condition name
(value) to a more legible one (value).

This is a **required** parameter in the configuration file, unless you are
processing resting-state data. If left as `None` and
[`task_is_rest`][mne_bids_pipeline._config.task_is_rest] is not `True`, we will raise an
error.

???+ example "Example"
    Specifying conditions as lists of strings:
    ```python
    conditions = ['auditory/left', 'visual/left']
    conditions = ['auditory/left', 'auditory/right']
    conditions = ['auditory']  # All "auditory" conditions (left AND right)
    conditions = ['auditory', 'visual']
    conditions = ['left', 'right']
    conditions = None  # for a resting-state analysis
    ```
    Pass a dictionary to define a mapping:
    ```python
    conditions = {'simple_name': 'complex/condition/with_subconditions'}
    conditions = {'correct': 'response/correct',
                  'incorrect': 'response/incorrect'}
"""

epochs_tmin: float = -0.2
"""
The beginning of an epoch, relative to the respective event, in seconds.

???+ example "Example"
    ```python
    epochs_tmin = -0.2  # 200 ms before event onset
    ```
"""

epochs_tmax: float = 0.5
"""
The end of an epoch, relative to the respective event, in seconds.
???+ example "Example"
    ```python
    epochs_tmax = 0.5  # 500 ms after event onset
    ```
"""

rest_epochs_duration: float | None = None
"""
Duration of epochs in seconds.
"""

rest_epochs_overlap: float | None = None
"""
Overlap between epochs in seconds. This is used if the task is `'rest'`
and when the annotations do not contain any stimulation or behavior events.
"""

baseline: tuple[float | None, float | None] | None = (None, 0)
"""
Specifies which time interval to use for baseline correction of epochs;
if `None`, no baseline correction is applied.

???+ example "Example"
    ```python
    baseline = (None, 0)  # beginning of epoch until time point zero
    ```
"""

# ## Artifact removal

# ### Stimulation artifact
#
# When using electric stimulation systems, e.g. for median nerve or index
# stimulation, it is frequent to have a stimulation artifact. This option
# allows to fix it by linear interpolation early in the pipeline on the raw
# data.

fix_stim_artifact: bool = False
"""
Apply interpolation to fix stimulation artifact.

???+ example "Example"
    ```python
    fix_stim_artifact = False
    ```
"""

stim_artifact_tmin: float = 0.0
"""
Start time of the interpolation window in seconds.

???+ example "Example"
    ```python
    stim_artifact_tmin = 0.  # on stim onset
    ```
"""

stim_artifact_tmax: float = 0.01
"""
End time of the interpolation window in seconds.

???+ example "Example"
    ```python
    stim_artifact_tmax = 0.01  # up to 10ms post-stimulation
    ```
"""

# ### SSP, ICA, and artifact regression

regress_artifact: dict[str, Any] | None = None
"""
Keyword arguments to pass to the `mne.preprocessing.EOGRegression` model used
in `mne.preprocessing.regress_artifact`. If `None`, no time-domain regression will
be applied. Note that any channels picked in `regress_artifact["picks_artifact"]` will
have the same time-domain filters applied to them as the experimental data.

Artifact regression is applied before SSP or ICA.

???+ example "Example"
    For example, if you have MEG reference channel data recorded in three
    miscellaneous channels, you could do:

    ```python
    regress_artifact = {
        "picks": "meg",
        "picks_artifact": ["MISC 001", "MISC 002", "MISC 003"]
    }
    ```
"""

spatial_filter: Literal["ssp", "ica"] | None = None
"""
Whether to use a spatial filter to detect and remove artifacts. The BIDS
Pipeline offers the use of signal-space projection (SSP) and independent
component analysis (ICA).

Use `'ssp'` for SSP, `'ica'` for ICA, and `None` if you do not wish to apply
a spatial filter for artifact removal.

The Pipeline will try to automatically discover EOG and ECG artifacts. For SSP,
it will then produce projection vectors that remove ("project out") these
artifacts from the data. For ICA, the independent components related to
EOG and ECG activity will be omitted during the signal reconstruction step in
order to remove the artifacts. The ICA procedure can be configured in various
ways using the configuration options you can find below.

!!! warning "ICA requires manual intervention!"
    After the automatic ICA component detection step, review each subject's
    `*_report.html` report file check if the set of ICA components to be removed
    is correct. Adjustments should be made to the `*_proc-ica_components.tsv`
    file, which will then be used in the step that is applied during ICA.

    ICA component order can be considered arbitrary, so any time the ICA is
    re-fit – i.e., if you change any parameters that affect steps prior to
    ICA fitting – this file will need to be updated!
"""

process_raw_clean: bool = True
"""
Whether to apply the spatial filter to the raw data to produce `_proc-clean_raw.fif`
files. If `False`, only the epochs will be cleaned, which can save on processing time,
disk space, and report length. Regardless of this setting, necessary intermediate
processed raw files like `_proc-filt_raw.fif` and similar files will be saved to disk.
"""

min_ecg_epochs: Annotated[int, Ge(1)] = 5
"""
Minimal number of ECG epochs needed to compute SSP projectors.
"""

min_eog_epochs: Annotated[int, Ge(1)] = 5
"""
Minimal number of EOG epochs needed to compute SSP projectors.
"""

n_proj_eog: dict[str, float] = dict(n_mag=1, n_grad=1, n_eeg=1)
"""
Number of SSP vectors to create for EOG artifacts for each channel type.
"""

n_proj_ecg: dict[str, float] = dict(n_mag=1, n_grad=1, n_eeg=1)
"""
Number of SSP vectors to create for ECG artifacts for each channel type.
"""

ecg_proj_from_average: bool = True
"""
Whether to calculate the ECG projection vectors based on the the averaged or
on individual ECG epochs.
"""

eog_proj_from_average: bool = True
"""
Whether to calculate the EOG projection vectors based on the the averaged or
on individual EOG epochs.
"""

ssp_meg: Literal["separate", "combined", "auto"] = "auto"
"""
Whether to compute SSP vectors for MEG channels separately (`'separate'`)
or jointly (`'combined'`) for magnetometers and gradiomenters. When using
Maxwell filtering, magnetometer and gradiometer signals are synthesized from
multipole moments jointly and are no longer independent, so it can be useful to
estimate projectors from all MEG sensors simultaneously. The default is
`'auto'`, which will use `'combined'` when Maxwell filtering is used and
`'separate'` otherwise.
"""

ssp_reject_ecg: dict[str, float] | Literal["autoreject_global"] | None = None
"""
Peak-to-peak amplitude limits of the ECG epochs to exclude from SSP fitting.
This allows you to remove strong transient artifacts, which could negatively
affect SSP performance.

The pipeline will automatically try to detect ECG artifacts in
your data, and remove them via SSP. For this to work properly, it is
recommended to **not** specify rejection thresholds for ECG channels here –
otherwise, SSP won't be able to "see" these artifacts.
???+ example "Example"
    ```python
    ssp_reject_ecg = {'grad': 10e-10, 'mag': 20e-12, 'eeg': 400e-6}
    ssp_reject_ecg = {'grad': 15e-10}
    ssp_reject_ecg = None
    ```
"""

ssp_reject_eog: dict[str, float] | Literal["autoreject_global"] | None = None
"""
Peak-to-peak amplitude limits of the EOG epochs to exclude from SSP fitting.
This allows you to remove strong transient artifacts, which could negatively
affect SSP performance.

The pipeline will automatically try to detect EOG artifacts in
your data, and remove them via SSP. For this to work properly, it is
recommended to **not** specify rejection thresholds for EOG channels here –
otherwise, SSP won't be able to "see" these artifacts.
???+ example "Example"
    ```python
    ssp_reject_eog = {'grad': 10e-10, 'mag': 20e-12, 'eeg': 400e-6}
    ssp_reject_eog = {'grad': 15e-10}
    ssp_reject_eog = None
    ```
"""

ssp_ecg_channel: str | dict[str, str] | None = None
"""
Channel to use for ECG SSP. Can be useful when the autodetected ECG channel
is not reliable. If `str`, the same channel will be used for all subjects.
If `dict`, possibly different channels will be used for each subject/session.
Dict values must be channel names, and dict keys must have the form `"sub-X"` (to use
the same channel for each session within a subject) or `"sub-X_ses-Y"` (to use possibly
different channels for each session of a given subject). Use dict key `"default"`
to set a default channel when using a dict.

???+ example "Example"
    Treat `T8` as virtual ECG channel:
    ```python
    ecg_channel = 'T8'
    ```

    Use `ECG`, but for sub-04, use MISC001 and for sub-05 session 1 use MISC002:
    ```python
    ssp_ecg_channel = dict()
    ssp_ecg_channel["default"] = 'ECG'
    ssp_ecg_channel['sub-04'] = 'MISC001'
    ssp_ecg_channel['sub-05_ses-1'] = 'MISC002'
    ```
    Note that `collections.defaultdict` cannot be used because it causes problems
    with pickling, which is used under the hood for caching and parallelization.
"""

ica_reject: dict[str, float] | Literal["autoreject_local"] | None = None
"""
Peak-to-peak amplitude limits to exclude epochs from ICA fitting. This allows you to
remove strong transient artifacts from the epochs used for fitting ICA, which could
negatively affect ICA performance.

The parameter values are the same as for [`reject`][mne_bids_pipeline._config.reject],
but `"autoreject_global"` is not supported. `"autoreject_local"` here behaves
differently, too: it is only used to exclude bad epochs from ICA fitting; we do not
perform any interpolation.

???+ info
    We don't support `"autoreject_global"` here (as opposed to
    [`reject`][mne_bids_pipeline._config.reject]) because in the past, we found that
    rejection thresholds were too strict before running ICA, i.e., too many epochs
    got rejected. `"autoreject_local"`, however, usually performed nicely.
    The `autoreject` documentation
    [recommends](https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html)
    running local `autoreject` before and after ICA, which can be achieved by setting
    both, `ica_reject` and [`reject`][mne_bids_pipeline._config.reject], to
    `"autoreject_local"`.

If passing a dictionary, the rejection limits will also be applied to the ECG and EOG
epochs created to find heart beats and ocular artifacts.

???+ info
    MNE-BIDS-Pipeline will automatically try to detect EOG and ECG artifacts in
    your data, and remove them. For this to work properly, it is recommended
    to **not** specify rejection thresholds for EOG and ECG channels here –
    otherwise, ICA won't be able to "see" these artifacts.

???+ info
    This setting is applied only to the epochs that are used for **fitting** ICA. The
    goal is to make it easier for ICA to produce a good decomposition. After fitting,
    ICA is applied to the epochs to be analyzed, usually with one or more components
    removed (as to remove artifacts). But even after ICA cleaning, some epochs may still
    contain large-amplitude artifacts. Those epochs can then be rejected by using
    the [`reject`][mne_bids_pipeline._config.reject] parameter.

???+ example "Example"
    ```python
    ica_reject = {'grad': 10e-10, 'mag': 20e-12, 'eeg': 400e-6}
    ica_reject = {'grad': 15e-10}
    ica_reject = None  # no rejection before fitting ICA
    ica_reject = "autoreject_global"  # find global (per channel type) PTP thresholds before fitting ICA
    ica_reject = "autoreject_local"  # find local (per channel) thresholds and repair epochs before fitting ICA
    ```
"""  # noqa: E501

ica_algorithm: Literal[
    "picard", "fastica", "extended_infomax", "picard-extended_infomax"
] = "picard"
"""
The ICA algorithm to use. `"picard-extended_infomax"` operates `picard` such that the
generated ICA decomposition is identical to the one generated by the extended Infomax
algorithm (but may converge in less time).
"""

ica_l_freq: float | None = 1.0
"""
The cutoff frequency of the high-pass filter to apply before running ICA.
Using a relatively high cutoff like 1 Hz will remove slow drifts from the
data, yielding improved ICA results. Must be set to 1 Hz or above.

Set to `None` to not apply an additional high-pass filter.

!!! info
      The filter will be applied to raw data which was already filtered
      according to the `l_freq` and `h_freq` settings. After filtering, the
      data will be epoched, and the epochs will be submitted to ICA.

!!! info
    The Pipeline will only allow you to perform ICA on data that has been
    high-pass filtered with a 1 Hz cutoff or higher. This is a conscious,
    opinionated (but partially data-driven) decision made by the developers.
    If you have reason to challenge this behavior, please get in touch with
    us so we can discuss.
"""

ica_h_freq: float | None = None
"""
The cutoff frequency of the low-pass filter to apply before running ICA.
"""

ica_max_iterations: int = 500
"""
Maximum number of iterations to decompose the data into independent
components. A low number means to finish earlier, but the consequence is
that the algorithm may not have finished converging. To ensure
convergence, pick a high number here (e.g. 3000); yet the algorithm will
terminate as soon as it determines that is has successfully converged, and
not necessarily exhaust the maximum number of iterations. Note that the
default of 200 seems to be sufficient for Picard in many datasets, because
it converges quicker than the other algorithms; but e.g. for FastICA, this
limit may be too low to achieve convergence.
"""

ica_n_components: float | int | None = None
"""
MNE conducts ICA as a sort of a two-step procedure: First, a PCA is run
on the data (trying to exclude zero-valued components in rank-deficient
data); and in the second step, the principal components are passed
to the actual ICA. You can select how many of the total principal
components to pass to ICA – it can be all or just a subset. This determines
how many independent components to fit, and can be controlled via this
setting.

If int, specifies the number of principal components that are passed to the
ICA algorithm, which will be the number of independent components to
fit. It must not be greater than the rank of your data (which is typically
the number of channels, but may be less in some cases).

If float between 0 and 1, all principal components with cumulative
explained variance less than the value specified here will be passed to
ICA.

If `None` (default), `0.999999` will be used to avoid issues when working with
rank-deficient data.

This setting may drastically alter the time required to compute ICA.
"""

ica_decim: int | None = None
"""
The decimation parameter to compute ICA. If 5 it means
that 1 every 5 sample is used by ICA solver. The higher the faster
it is to run but the less data you have to compute a good ICA. Set to
`1` or `None` to not perform any decimation.
"""

ica_use_ecg_detection: bool = True
"""
Whether to use the MNE ECG detection on the ICA components.
"""

ica_ecg_threshold: float = 0.1
"""
The cross-trial phase statistics (CTPS) threshold parameter used for detecting
ECG-related ICs.
"""

ica_use_eog_detection: bool = True
"""
Whether to use the MNE EOG detection on the ICA components.
"""

ica_eog_threshold: float = 3.0
"""
The threshold to use during automated EOG classification. Lower values mean
that more ICs will be identified as EOG-related. If too low, the
false-alarm rate increases dramatically.
"""

ica_use_icalabel: bool = False
"""
Whether to use MNE-ICALabel to automatically label ICA components. Only available for
EEG data.

!!! info
    Using MNE-ICALabel mandates that you also set:
    ```python
    eeg_reference = "average"
    ica_l_freq = 1
    ica_h_freq = 100
    ```
    It will also apply the average reference to the data before running or applying ICA.

!!! info
    Using this requires `mne-icalabel` package, which in turn will require you to
    install a suitable runtime (`onnxruntime` or `pytorch`).
"""

ica_icalabel_include: Annotated[
    UniqueSequence[
        Literal[
            "brain",
            "muscle artifact",
            "eye blink",
            "heart beat",
            "line noise",
            "channel noise",
            "other",
        ]
    ],
    Len(1, 7),
] = ("brain", "other")
"""
Which independent components (ICs) to keep based on the labels given by ICLabel.
Possible labels are:
```
["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]
```
Default behaviour: keeps all components except those with a label other then specified
IF they meet the default exclusion threshold of 0.8
"""  # noqa: E501

ica_exclusion_thresholds: dict[str, float] = {
    "brain": 0.8,
    "muscle artifact": 0.8,
    "eye blink": 0.8,
    "heart beat": 0.8,
    "line noise": 0.8,
    "channel noise": 0.8,
    "other": 0.8,
}
"""
ICLabel class minimum probability thresholds for excluding components.
You can set single values like `{"eye blink": 0.7,"brain": 0.8}` with the remaining
values being the default.

Each component gets a probability distribution over all classes e.g.
`[0.7, 0.1, 0.1, 0.05, 0.05, 0, 0]` (order as in the dict). Flags components to be
dropped IF they meet any of the exclusion thresholds of classes not in
`ica_icalabel_include`.
"""

ica_class_thresholds: dict[str, float] = {
    "brain": 0.3,
    "muscle artifact": 0.3,
    "eye blink": 0.3,
    "heart beat": 0.3,
    "line noise": 0.3,
    "channel noise": 0.3,
    "other": 0.3,
}
"""
ICLabel class minimum probability thresholds for considering components member of a
class (like "brain", etc). You can set single values like
`{"eye blink": 0.3,"brain": 0.3}` with the remaining values being the default.

Each component gets a probability distribution over all classes e.g.
`[0.7, 0.1, 0.1, 0, 0, 0, 0]` (order as in the dict). Makes sure components are kept IF
they meet any of the class thresholds of classes in `ica_icalabel_include`.
"""

# noqa: E501

# ### Amplitude-based artifact rejection
#
# ???+ info "Good Practice / Advice"
#     Have a look at your raw data and train yourself to detect a blink, a heart
#     beat and an eye movement.
#     You can do a quick average of blink data and check what the amplitude looks
#     like.

reject: dict[str, float] | Literal["autoreject_global", "autoreject_local"] | None = (
    None
)
"""
Peak-to-peak amplitude limits to mark epochs as bad. This allows you to remove
epochs with strong transient artifacts.

!!! info
      The rejection is performed **after** SSP or ICA, if any of those methods
      is used. To reject epochs **before** fitting ICA, see the
      [`ica_reject`][mne_bids_pipeline._config.ica_reject] setting.

If `None` (default), do not apply artifact rejection.

If a dictionary, manually specify rejection thresholds (see examples).
The thresholds provided here must be at least as stringent as those in
[`ica_reject`][mne_bids_pipeline._config.ica_reject] if using ICA. In case of
`'autoreject_global'`, thresholds for any channel that do not meet this
requirement will be automatically replaced with those used in `ica_reject`.

If `"autoreject_global"`, use [`autoreject`](https://autoreject.github.io) to find
suitable "global" rejection thresholds for each channel type, i.e., `autoreject`
will generate a dictionary with (hopefully!) optimal thresholds for each
channel type.

If `"autoreject_local"`, use "local" `autoreject` to detect (and potentially repair) bad
channels in each epoch.
Use [`autoreject_n_interpolate`][mne_bids_pipeline._config.autoreject_n_interpolate]
to control how many channels are allowed to be bad before an epoch gets dropped.

???+ example "Example"
    ```python
    reject = {"grad": 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
    reject = {"eeg": 100e-6, "eog": 250e-6}
    reject = None  # no rejection based on PTP amplitude
    reject = "autoreject_global"  # find global (per channel type) PTP thresholds
    reject = "autoreject_local"  # find local (per channel) thresholds and repair epochs
    ```
"""

reject_tmin: float | None = None
"""
Start of the time window used to reject epochs. If `None`, the window will
start with the first time point. Has no effect if
[`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

???+ example "Example"
    ```python
    reject_tmin = -0.1  # 100 ms before event onset.
    ```
"""

reject_tmax: float | None = None
"""
End of the time window used to reject epochs. If `None`, the window will end
with the last time point. Has no effect if
[`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

???+ example "Example"
    ```python
    reject_tmax = 0.3  # 300 ms after event onset.
    ```
"""

autoreject_n_interpolate: FloatArrayLike = [4, 8, 16]
"""
The maximum number of bad channels in an epoch that `autoreject` local will try to
interpolate. The optimal number among this list will be estimated using a
cross-validation procedure; this means that the more elements are provided here, the
longer the `autoreject` run will take. If the number of bad channels in an epoch
exceeds this value, the channels won't be interpolated and the epoch will be dropped.

!!! info
    This setting only takes effect if [`reject`][mne_bids_pipeline._config.reject] has
    been set to `"autoreject_local"`.

!!! info
    Channels marked as globally bad in the BIDS dataset (in `*_channels.tsv)`) will not
    be considered (i.e., will remain marked as bad and not analyzed by autoreject).
"""

# %%
# # Sensor-level analysis

# ## Condition contrasts

contrasts: Sequence[tuple[str, str] | ArbitraryContrast] = []
"""
The conditions to contrast via a subtraction of ERPs / ERFs. The list elements
can either be tuples or dictionaries (or a mix of both). Each element in the
list corresponds to a single contrast.

A tuple specifies a one-vs-one contrast, where the second condition is
subtracted from the first.

If a dictionary, must contain the following keys:

- `name`: a custom name of the contrast
- `conditions`: the conditions to contrast
- `weights`: the weights associated with each condition.

Pass an empty list to avoid calculation of any contrasts.

For the contrasts to be computed, the appropriate conditions must have been
epoched, and therefore the conditions should either match or be subsets of
`conditions` above.

???+ example "Example"
    Contrast the "left" and the "right" conditions by calculating
    `left - right` at every time point of the evoked responses:
    ```python
    contrasts = [('left', 'right')]  # Note we pass a tuple inside the list!
    ```

    Contrast the "left" and the "right" conditions within the "auditory" and
    the "visual" modality, and "auditory" vs "visual" regardless of side:
    ```python
    contrasts = [('auditory/left', 'auditory/right'),
                 ('visual/left', 'visual/right'),
                 ('auditory', 'visual')]
    ```

    Contrast the "left" and the "right" regardless of side, and compute an
    arbitrary contrast with a gradient of weights:
    ```python
    contrasts = [
        ('auditory/left', 'auditory/right'),
        {
            'name': 'gradedContrast',
            'conditions': [
                'auditory/left',
                'auditory/right',
                'visual/left',
                'visual/right'
            ],
            'weights': [-1.5, -.5, .5, 1.5]
        }
    ]
    ```
"""

# ## Decoding / MVPA

decode: bool = True
"""
Whether to perform decoding (MVPA) on the specified
[`contrasts`][mne_bids_pipeline._config.contrasts]. Classifiers will be trained
on entire epochs ("full-epochs decoding"), and separately on each time point
("time-by-time decoding"), trying to learn how to distinguish the contrasting
conditions.
"""

decoding_which_epochs: Literal["uncleaned", "after_ica", "after_ssp", "cleaned"] = (
    "cleaned"
)
"""
This setting controls which epochs will be fed into the decoding algorithms.

!!! info
    Decoding is a very powerful tool that often can deal with noisy data surprisingly
    well. Depending on the specific type of data, artifacts, and analysis performed,
    decoding performance may even improve with less pre-processed data, as
    processing steps such as ICA or SSP often remove parts of the signal, too, in
    addition to noise. By default, MNE-BIDS-Pipeline uses cleaned epochs for decoding,
    but you may choose to use entirely uncleaned epochs, or epochs before the final
    PTP-based rejection or Autoreject step.

!!! info
    No other sensor- and source-level processing steps will be affected by this setting
    and use cleaned epochs only.

If `"uncleaned"`, use the "raw" epochs before any ICA / SSP, PTP-based, or Autoreject
cleaning (epochs with the filename `*_epo.fif`, without a `proc-` part).

If `"after_ica"` or `"after_ssp"`, use the epochs that were cleaned via ICA or SSP, but
before a followup cleaning through PTP-based rejection or Autorejct (epochs with the
filename `*proc-ica_epo.fif` or `*proc-ssp_epo.fif`).

If `"cleaned"`, use the epochs after ICA / SSP and the following cleaning through
PTP-based rejection or Autoreject (epochs with the filename `*proc-clean_epo.fif`).
"""

decoding_epochs_tmin: float | None = 0.0
"""
The first time sample to use for full epochs decoding. By default it starts
at 0. If `None`,, it starts at the beginning of the epoch. Does not affect time-by-time
decoding.
"""

decoding_epochs_tmax: float | None = None
"""
The last time sample to use for full epochs decoding. By default it is set
to None so it ends at the end of the epoch.
"""

decoding_metric: str = "roc_auc"
"""
The metric to use for estimating classification performance. It can be
`'roc_auc'` or `'accuracy'` – or any other metric supported by `scikit-learn`.

With ROC AUC, chance level is the same regardless of class balance, that is,
you don't need to be worried about **exactly** balancing class sizes.
"""

decoding_n_splits: Annotated[int, Ge(2)] = 5
"""
The number of folds (also called "splits") to use in the K-fold cross-validation
scheme.
"""

decoding_time_generalization: bool = False
"""
Whether to perform time generalization.

Time generalization (also called "temporal generalization" or "generalization
across time", GAT) is an extension of the time-by-time decoding approach.
Again, a separate classifier is trained on each time point. But instead of just
testing the model on the same time point in the test data, it will be tested
on **all** time points.

!!! cite ""
    [T]he manner in which the trained classifiers generalize across time, and
    from one experimental condition to another, sheds light on the temporal
    organization of information-processing stages.

    [DOI: 10.1016/j.tics.2014.01.002](https://doi.org/10.1016/j.tics.2014.01.002)

Because each classifier is trained and tested on **all** time points, this
procedure may take a significant amount of time.
"""

decoding_time_generalization_decim: int = 1
"""
Says how much to decimate data before time generalization decoding.
This is done in addition to the decimation done at the epochs level via the
[`epochs_decim`][mne_bids_pipeline._config.epochs_decim] parameter. This can be
used to greatly speed up time generalization at the cost of lower time
resolution in the resulting matrix.
"""

decoding_csp: bool = False
"""
Whether to run decoding via Common Spatial Patterns (CSP) analysis on the
data. CSP takes as input data covariances that are estimated on different
time and frequency ranges. This allows to obtain decoding scores defined over
time and frequency.
"""

decoding_csp_times: FloatArrayLike | None = None
"""
The edges of the time bins to use for CSP decoding.
Must contain at least two elements. By default, 5 equally-spaced bins are
created across the non-negative time range of the epochs.
All specified time points must be contained in the epochs interval.
If an empty list, do not perform **time-frequency** analysis, and only run CSP on
**frequency** data.

???+ example "Example"
    Create 3 equidistant time bins (0–0.2, 0.2–0.4, 0.4–0.6 sec):
    ```python
    decoding_csp_times = np.linspace(start=0, stop=0.6, num=4)
    ```
    Create 2 time bins of different durations (0–0.4, 0.4–0.6 sec):
    ```python
    decoding_csp_times = [0, 0.4, 0.6]
    ```
"""

decoding_csp_freqs: dict[str, FloatArrayLike] | None = None
"""
The edges of the frequency bins to use for CSP decoding.

This parameter must be a dictionary with:
- keys specifying the unique identifier or "name" to use for the frequency
  range to be treated jointly during statistical testing (such as "alpha" or
  "beta"), and
- values must be list-like objects containing at least two scalar values,
  specifying the edges of the respective frequency bin(s), e.g., `[8, 12]`.

Defaults to two frequency bins, one from
[`time_frequency_freq_min`][mne_bids_pipeline._config.time_frequency_freq_min]
to the midpoint between this value and
[`time_frequency_freq_max`][mne_bids_pipeline._config.time_frequency_freq_max];
and the other from that midpoint to `time_frequency_freq_max`.
???+ example "Example"
    Create two frequency bins, one for 4–8 Hz, and another for 8–14 Hz, which
    will be clustered together during statistical testing (in the
    time-frequency plane):
    ```python
    decoding_csp_freqs = {
        'custom_range': [4, 8, 14]
    }
    ```
    Create the same two frequency bins, but treat them separately during
    statistical testing (i.e., temporal clustering only):
    ```python
    decoding_csp_freqs = {
        'theta': [4, 8],
        'alpha': [8, 14]
    }
    ```
    Create 5 equidistant frequency bins from 4 to 14 Hz:
    ```python
    decoding_csp_freqs = {
        'custom_range': np.linspace(
            start=4,
            stop=14,
            num=5+1  # We need one more to account for the endpoint!
        )
    }
"""

n_boot: int = 5000
"""
The number of bootstrap resamples when estimating the standard error and
confidence interval of the mean decoding scores.
"""

cluster_forming_t_threshold: float | None = None
"""
The t-value threshold to use for forming clusters in the cluster-based
permutation test run on the the time-by-time decoding scores.
Data points with absolute t-values greater than this value
will be used to form clusters. If `None`, the threshold will be automatically
determined to correspond to a p-value of 0.05 for the given number of
participants in a one-tailed test.

!!! info
    Only points with the same sign will be clustered together.
"""

cluster_n_permutations: int = 10_000
"""
The maximum number of permutations to perform in a cluster-based permutation
test to determine the significance of the decoding scores across participants.
"""

cluster_permutation_p_threshold: Annotated[float, Interval(gt=0, lt=1)] = 0.05
"""
The alpha level (p-value, p threshold) to use for rejecting the null hypothesis
that the clusters show no significant difference between conditions. This is
used in the permutation test which takes place after forming the clusters.

!!! info
    To control how clusters are formed, see
    [`cluster_forming_t_threshold`][mne_bids_pipeline._config.cluster_forming_t_threshold].
"""

# ## Time-frequency analysis

time_frequency_conditions: Sequence[str] = []
"""
The conditions to compute time-frequency decomposition on.

???+ example "Example"
    ```python
    time_frequency_conditions = ['left', 'right']
    ```
"""

time_frequency_freq_min: float | None = 8
"""
Minimum frequency for the time frequency analysis, in Hz.
???+ example "Example"
    ```python
    time_frequency_freq_min = 0.3  # 0.3 Hz
    ```
"""

time_frequency_freq_max: float | None = 40
"""
Maximum frequency for the time frequency analysis, in Hz.
???+ example "Example"
    ```python
    time_frequency_freq_max = 22.3  # 22.3 Hz
    ```
"""

time_frequency_cycles: float | FloatArrayLike | None = None
"""
The number of cycles to use in the Morlet wavelet. This can be a single number
or one per frequency, where frequencies are calculated via
`np.arange(time_frequency_freq_min, time_frequency_freq_max)`.
If `None`, uses
`np.arange(time_frequency_freq_min, time_frequency_freq_max) / 3`.
"""

time_frequency_subtract_evoked: bool = False
"""
Whether to subtract the evoked response (averaged across all epochs) from the
epochs before passing them to time-frequency analysis. Set this to `True` to
highlight induced activity.

!!! info
     This also applies to CSP analysis.
"""

time_frequency_baseline: tuple[float, float] | None = None
"""
Baseline period to use for the time-frequency analysis. If `None`, no baseline.
???+ example "Example"
    ```python
    time_frequency_baseline = (None, 0)
    ```
"""

time_frequency_baseline_mode: str = "mean"
"""
Baseline mode to use for the time-frequency analysis. Can be chosen among:
"mean" or "ratio" or "logratio" or "percent" or "zscore" or "zlogratio".
???+ example "Example"
    ```python
    time_frequency_baseline_mode = 'mean'
    ```
"""

time_frequency_crop: dict[str, float] | None = None
"""
Period and frequency range to crop the time-frequency analysis to.
If `None`, no cropping.

???+ example "Example"
    ```python
    time_frequency_crop = dict(tmin=-0.3, tmax=0.5, fmin=5., fmax=20.)
    ```
"""

# ## Group-level analysis

interpolate_bads_grand_average: bool = True
"""
Interpolate bad sensors in each dataset before calculating the grand
average. This parameter is passed to the `mne.grand_average` function via
the keyword argument `interpolate_bads`. It requires to have channel
locations set.

???+ example "Example"
    ```python
    interpolate_bads_grand_average = True
    ```
"""

# %%
# # Source-level analysis

# ## General source analysis settings

run_source_estimation: bool = True
"""
Whether to run source estimation processing steps if not explicitly requested.
"""

# ## BEM surface

use_template_mri: str | None = None
"""
Whether to use a template MRI subject such as FreeSurfer's `fsaverage` subject.
This may come in handy if you don't have individual MR scans of your
participants, as is often the case in EEG studies.

Note that the template MRI subject must be available as a subject
in your subjects_dir. You can use for example a scaled version
of fsaverage that could get with
[`mne.scale_mri`](https://mne.tools/stable/generated/mne.scale_mri.html).
Scaling fsaverage can be a solution to problems that occur when the head of a
subject is small compared to `fsaverage` and, therefore, the default
coregistration mislocalizes MEG sensors inside the head.

???+ example "Example"
    ```python
    use_template_mri = "fsaverage"
    ```
"""

adjust_coreg: bool = False
"""
Whether to adjust the coregistration between the MRI and the channels
locations, possibly combined with the digitized head shape points.
Setting it to True is mandatory if you use a template MRI subject
that is different from `fsaverage`.

???+ example "Example"
    ```python
    adjust_coreg = True
    ```
"""

bem_mri_images: Literal["FLASH", "T1", "auto"] = "auto"
"""
Which types of MRI images to use when creating the BEM model.
If `'FLASH'`, use FLASH MRI images, and raise an exception if they cannot be
found.

???+ info "Advice"
    It is recommended to use the FLASH images if available, as the quality
    of the extracted BEM surfaces will be higher.

If `'T1'`, create the BEM surfaces from the T1-weighted images using the
`watershed` algorithm.

If `'auto'`, use FLASH images if available, and use the `watershed``
algorithm with the T1-weighted images otherwise.

*[FLASH MRI]: Fast low angle shot magnetic resonance imaging
"""

recreate_bem: bool = False
"""
Whether to re-create the BEM surfaces, even if existing surfaces have been
found. If `False`, the BEM surfaces are only created if they do not exist
already. `True` forces their recreation, overwriting existing BEM surfaces.
"""

recreate_scalp_surface: bool = False
"""
Whether to re-create the scalp surfaces used for visualization of the
coregistration in the report and the lower-density coregistration surfaces.
If `False`, the scalp surface is only created if it does not exist already.
If `True`, forces a re-computation.
"""

freesurfer_verbose: bool = False
"""
Whether to print the complete output of FreeSurfer commands. Note that if
`False`, no FreeSurfer output might be displayed at all!"""

# ## Source space & forward solution

mri_t1_path_generator: Callable[[BIDSPath], BIDSPath] | None = None
"""
To perform source-level analyses, the Pipeline needs to generate a
transformation matrix that translates coordinates from MEG and EEG sensor
space to MRI space, and vice versa. This process, called "coregistration",
requires access to both, the electrophyisiological recordings as well as
T1-weighted MRI images of the same participant. If both are stored within
the same session, the Pipeline (or, more specifically, MNE-BIDS) can find the
respective files automatically.

However, in certain situations, this is not possible. Examples include:

- MRI was conducted during a different session than the electrophysiological
  recording.
- MRI was conducted in a single session, while electrophysiological recordings
  spanned across several sessions.
- MRI and electrophysiological data are stored in separate BIDS datasets to
  allow easier storage and distribution in certain situations.

To allow the Pipeline to find the correct MRI images and perform coregistration
automatically, we provide a "hook" that allows you to provide a custom
function whose output tells the Pipeline where to find the T1-weighted image.

The function is expected to accept a single parameter: The Pipeline will pass
a `BIDSPath` with the following parameters set based on the currently processed
electrophysiological data:

- the subject ID, `BIDSPath.subject`
- the experimental session, `BIDSPath.session`
- the BIDS root, `BIDSPath.root`

This `BIDSPath` can then be modified – or an entirely new `BIDSPath` can be
generated – and returned by the function, pointing to the T1-weighted image.

!!! info
    The function accepts and returns a single `BIDSPath`.

???+ example "Example"
    The MRI session is different than the electrophysiological session:
    ```python
    def get_t1_from_meeg(bids_path):
        bids_path.session = 'MRI'
        return bids_path


    mri_t1_path_generator = get_t1_from_meeg
    ```

    The MRI recording is stored in a different BIDS dataset than the
    electrophysiological data:
    ```python
    def get_t1_from_meeg(bids_path):
        bids_path.root = '/data/mri'
        return bids_path


    mri_t1_path_generator = get_t1_from_meeg
    ```
"""

mri_landmarks_kind: Callable[[BIDSPath], str] | None = None
"""
This config option allows to look for specific landmarks in the json
sidecar file of the T1 MRI file. This can be useful when we have different
fiducials coordinates e.g. the manually positioned fiducials or the
fiducials derived for the coregistration transformation of a given session.

???+ example "Example"
    We have one MRI session and we have landmarks with a kind
    indicating how to find the landmarks for each session:

    ```python
    def mri_landmarks_kind(bids_path):
        return f"ses-{bids_path.session}"
    ```
"""

spacing: Literal["oct5", "oct6", "ico4", "ico5", "all"] | int = "oct6"
"""
The spacing to use. Can be `'ico#'` for a recursively subdivided
icosahedron, `'oct#'` for a recursively subdivided octahedron,
`'all'` for all points, or an integer to use approximate
distance-based spacing (in mm). See (the respective MNE-Python documentation)
[https://mne.tools/dev/overview/cookbook.html#setting-up-the-source-space]
for more info.
"""

mindist: float = 5
"""
Exclude points closer than this distance (mm) to the bounding surface.
"""

# ## Inverse solution

loose: Annotated[float, Interval(ge=0, le=1)] | Literal["auto"] = 0.2
"""
A value between 0 and 1 that weights the source variances of the dipole components
that are parallel (tangential) to the cortical surface.

If `0`, then the inverse solution is computed with **fixed orientation**, i.e.,
only dipole components perpendicular to the cortical surface are considered.

If `1`, it corresponds to **free orientation**, i.e., dipole components with any
orientation are considered.

The default value, `0.2`, is suitable for surface-oriented source spaces.

For volume or mixed source spaces, choose `1.0`.

!!! info
    Support for modeling volume and mixed source spaces will be added in a future
    version of MNE-BIDS-Pipeline.
"""

depth: Annotated[float, Interval(ge=0, le=1)] | dict[str, Any] = 0.8
"""
If a number, it acts as the depth weighting exponent to use
(must be between `0` and`1`), with`0` meaning no depth weighting is performed.

Can also be a dictionary containing additional keyword arguments to pass to
`mne.forward.compute_depth_prior` (see docstring for details and defaults).
"""

inverse_method: Literal["MNE", "dSPM", "sLORETA", "eLORETA"] = "dSPM"
"""
Use minimum norm, dSPM (default), sLORETA, or eLORETA to calculate the inverse
solution.
"""

noise_cov: (
    tuple[float | None, float | None]
    | Literal["emptyroom", "rest", "ad-hoc"]
    | Callable[[BIDSPath], Covariance]
) = (None, 0)
"""
Specify how to estimate the noise covariance matrix, which is used in
inverse modeling.

If a tuple, it takes the form `(tmin, tmax)` with the time specified in
seconds. If the first value of the tuple is `None`, the considered
period starts at the beginning of the epoch. If the second value of the
tuple is `None`, the considered period ends at the end of the epoch.
The default, `(None, 0)`, includes the entire period before the event,
which is typically the pre-stimulus period.

If `'emptyroom'`, the noise covariance matrix will be estimated from an
empty-room MEG recording. The empty-room recording will be automatically
selected based on recording date and time. This cannot be used with EEG data.

If `'rest'`, the noise covariance will be estimated from a resting-state
recording (i.e., a recording with `task-rest` and without a `run` in the
filename).

If `'ad-hoc'`, a diagonal ad-hoc noise covariance matrix will be used.

You can also pass a function that accepts a `BIDSPath` and returns an
`mne.Covariance` instance. The `BIDSPath` will point to the file containing
the generated cleaned epochs data.

???+ example "Example"
    Use the period from start of the epoch until 100 ms before the experimental
    event:
    ```python
    noise_cov = (None, -0.1)
    ```

    Use the time period from the experimental event until the end of the epoch:
    ```python
    noise_cov = (0, None)
    ```

    Use an empty-room recording:
    ```python
    noise_cov = 'emptyroom'
    ```

    Use a resting-state recording:
    ```python
    noise_cov = 'rest'
    ```

    Use an ad-hoc covariance:
    ```python
    noise_cov = 'ad-hoc'
    ```

    Use a custom covariance derived from raw data:
    ```python
    def noise_cov(bids_path):
        bp = bids_path.copy().update(task="rest", run=None)
        raw_rest = mne_bids.read_raw_bids(bp)
        raw_rest.crop(tmin=5, tmax=60)
        cov = mne.compute_raw_covariance(raw_rest, rank="info")
        return cov
    ```
"""  # noqa: E501

noise_cov_method: Literal[
    "shrunk",
    "empirical",
    "diagonal_fixed",
    "oas",
    "ledoit_wolf",
    "factor_analysis",
    "shrinkage",
    "pca",
    "auto",
] = "shrunk"
"""
The noise covariance estimation method to use. See the MNE-Python documentation
of `mne.compute_covariance` for details.
"""

cov_rank: Literal["info"] | dict[str, Any] = "info"
"""
Specifies how to determine the rank of the data and associated noise covariance.
This is used when computing an inverse operator and when preprocessing data for
decoding. If set to `"info"` (default), the rank will be computed from the measurement
information. If it's a `dict`, the rank will be computed from the data used to
compute the covariance, with the `cov_rank` dict passed as keyword arguments
as `mne.compute_rank(inst, info=info, **cov_rank)` (where the `inst` and `info` will
automatically be determined by the `noise_cov` type).

???+ example "Example"
    Compute the rank from the data:
    ```python
    cov_rank = {"tol_kind": "relative", "tol": 1e-4}
    ```
"""

source_info_path_update: dict[str, str] | None = None
"""
When computing the forward and inverse solutions, it is important to
provide the `mne.Info` object from the data on which the noise covariance was
computed, to avoid problems resulting from mismatching ranks.
This parameter allows you to explicitly specify from which file to retrieve the
`mne.Info` object. Use this parameter to supply a dictionary to
`BIDSPath.update()` during the forward and inverse processing steps.
If set to `None` (default), the info will be retrieved either from the raw
file specified in `noise_cov`, or the cleaned evoked
(if `noise_cov` is None or `ad-hoc`).

???+ example "Example"
    Use the `Info` object stored in the cleaned epochs:
    ```python
    source_info_path_update = {'processing': 'clean',
                               'suffix': 'epo'}
    ```

    Use the `Info` object stored in a raw file (e.g. resting state):
    ```python
    source_info_path_update = {'processing': 'clean',
                                'suffix': 'raw',
                                'task': 'rest'}
    ```
    If you set `noise_cov = 'rest'` and `source_path_info = None`,
    then the behavior is identical to that above
    (it will automatically use the resting state data).

"""

inverse_targets: list[Literal["evoked"]] = ["evoked"]
"""
On which data to apply the inverse operator. Currently, the only supported
target is `'evoked'`. If no inverse computation should be done, pass an
empty list, `[]`.

???+ example "Example"
    Compute the inverse solution on evoked data:
    ```python
    inverse_targets = ['evoked']
    ```

    Don't compute an inverse solution:
    ```python
    inverse_targets = []
    ```
"""

# %%
# # Reports

# ## Report generation

report_evoked_n_time_points: int | None = None
"""
Specifies the number of time points to display for each evoked
in the report. If `None`, it defaults to the current default in MNE-Python.

???+ example "Example"
    Only display 5 time points per evoked
    ```python
    report_evoked_n_time_points = 5
    ```
"""

report_stc_n_time_points: int | None = None
"""
Specifies the number of time points to display for each source estimates
in the report. If `None`, it defaults to the current default in MNE-Python.

???+ example "Example"
    Only display 5 images per source estimate:
    ```python
    report_stc_n_time_points = 5
    ```
"""

report_add_epochs_image_kwargs: dict[str, Any] | None = None
"""
Specifies the limits for the color scales of the epochs_image in the report.
If `None`, it defaults to the current default in MNE-Python.

???+ example "Example"
    Set vmin and vmax to the epochs rejection thresholds (with unit conversion):

    ```python
    report_add_epochs_image_kwargs = {
        "grad": {"vmin": 0, "vmax": 1e13 * reject["grad"]},  # fT/cm
        "mag": {"vmin": 0, "vmax": 1e15 * reject["mag"]},  # fT
    }
    ```
"""

# %%
# # Caching
#
# Per default, the pipeline output is cached (temporarily stored),
# to avoid unnecessary reruns of previously computed steps.
# Yet, for consistency, changes in configuration parameters trigger
# automatic reruns of previous steps.
# !!! info
#     To force rerunning a given step, run the pipeline with the option: `--no-cache`.

memory_location: PathLike | bool | None = True
"""
If not None (or False), caching will be enabled and the cache files will be
stored in the given directory. The default (True) will use a
`"_cache"` subdirectory (name configurable via the
[`memory_subdir`][mne_bids_pipeline._config.memory_subdir]
variable) in the BIDS derivative root of the dataset.
"""

memory_subdir: str = "_cache"
"""
The caching directory name to use if `memory_location` is `True`.
"""

memory_file_method: Literal["mtime", "hash"] = "mtime"
"""
The method to use for cache invalidation (i.e., detecting changes). Using the
"modified time" reported by the filesystem (`'mtime'`, default) is very fast
but requires that the filesystem supports proper mtime reporting. Using file
hashes (`'hash'`) is slower and requires reading all input files but should
work on any filesystem.
"""

memory_verbose: int = 0
"""
The verbosity to use when using memory. The default (0) does not print, while
1 will print the function calls that will be cached. See the documentation for
the joblib.Memory class for more information."""

# %%
# # Parallelization
#
# These options control parallel processing (e.g., multiple subjects at once),

n_jobs: int = 1
"""
Specifies how many subjects you want to process in parallel. If `1`, disables
parallel processing.
"""

parallel_backend: Literal["loky", "dask"] = "loky"
"""
Specifies which backend to use for parallel job execution. `loky` is the
default backend used by `joblib`. `dask` requires [`Dask`](https://dask.org) to
be installed. Ignored if [`n_jobs`][mne_bids_pipeline._config.n_jobs] is set to
`1`.
"""

dask_open_dashboard: bool = False
"""
Whether to open the Dask dashboard in the default webbrowser automatically.
Ignored if `parallel_backend` is not `'dask'`.
"""

dask_temp_dir: PathLike | None = None
"""
The temporary directory to use by Dask. Dask places lock-files in this
directory, and also uses it to "spill" RAM contents to disk if the amount of
free memory in the system hits a critical low. It is recommended to point this
to a location on a fast, local disk (i.e., not a network-attached storage) to
ensure good performance. The directory needs to be writable and will be created
if it does not exist.

If `None`, will use `.dask-worker-space` inside of
[`deriv_root`][mne_bids_pipeline._config.deriv_root].
"""

dask_worker_memory_limit: str = "10G"
"""
The maximum amount of RAM per Dask worker.
"""

# %%
# # Logging
#
# These options control how much logging output is produced.

log_level: Literal["info", "error"] = "info"
"""
Set the pipeline logging verbosity.
"""

mne_log_level: Literal["info", "error"] = "error"
"""
Set the MNE-Python logging verbosity.
"""


# %%
# # Error handling
#
# These options control how errors while processing the data or the configuration file
# are handled.

on_error: Literal["continue", "abort", "debug"] = "abort"
"""
Whether to abort processing as soon as an error occurs, continue with all other
processing steps for as long as possible, or drop you into a debugger in case
of an error.

!!! info
    Enabling debug mode deactivates parallel processing.
"""

config_validation: Literal["raise", "warn", "ignore"] = "raise"
"""
How strictly to validate the configuration. Errors are always raised for
invalid entries (e.g., not providing `ch_types`). This setting controls
how to handle *possibly* or *likely* incorrect entries, such as likely
misspellings (e.g., providing `session` instead of `sessions`).
"""
