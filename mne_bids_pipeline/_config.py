"""Default settings for data processing and analysis.
"""

from typing import (
    Optional, Union, Iterable, List, Tuple, Dict, Callable, Literal)

from numpy.typing import ArrayLike

import mne
from mne_bids import BIDSPath
import numpy as np

from mne_bids_pipeline.typing import PathLike, ArbitraryContrast


###############################################################################
# Config parameters
# -----------------

study_name: str = ''
"""
Specify the name of your study. It will be used to populate filenames for
saving the analysis results.

???+ example "Example"
    ```python
    study_name = 'my-study'
    ```
"""

bids_root: Optional[PathLike] = None
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

deriv_root: Optional[PathLike] = None
"""
The root of the derivatives directory in which the pipeline will store
the processing results. If `None`, this will be
`derivatives/mne-bids-pipeline` inside the BIDS root.

Note: Note
    If specified and you wish to run the source analysis steps, you must
    set [`subjects_dir`][mne_bids_pipeline._config.subjects_dir] as well.
"""

subjects_dir: Optional[PathLike] = None
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

Note: Note
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

Note: Note
    Enabling interactive mode deactivates parallel processing.
"""

sessions: Union[List, Literal['all']] = 'all'
"""
The sessions to process. If `'all'`, will process all sessions found in the
BIDS dataset.
"""

task: str = ''
"""
The task to process.
"""

runs: Union[Iterable, Literal['all']] = 'all'
"""
The runs to process. If `'all'`, will process all runs found in the
BIDS dataset.
"""

exclude_runs: Optional[Dict[str, List[str]]] = None
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

crop_runs: Optional[Tuple[float, float]] = None
"""
Crop the raw data of each run to the specified time interval `[tmin, tmax]`,
in seconds. The runs will be cropped before Maxwell or frequency filtering is
applied. If `None`, do not crop the data.
"""

acq: Optional[str] = None
"""
The BIDS `acquisition` entity.
"""

proc: Optional[str] = None
"""
The BIDS `processing` entity.
"""

rec: Optional[str] = None
"""
The BIDS `recording` entity.
"""

space: Optional[str] = None
"""
The BIDS `space` entity.
"""

plot_psd_for_runs: Union[Literal['all'], Iterable[str]] = 'all'
"""
For which runs to add a power spectral density (PSD) plot to the generated
report. This can take a considerable amount of time if you have many long
runs. In this case, specify the runs, or pass an empty list to disable raw PSD
plotting.
"""

subjects: Union[Iterable[str], Literal['all']] = 'all'
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

exclude_subjects: Iterable[str] = []
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

ch_types: Iterable[Literal['meg', 'mag', 'grad', 'eeg']] = []
"""
The channel types to consider.

!!! info
    Currently, MEG and EEG data cannot be processed together.

???+ example "Example"
    ```python
    # Use EEG channels:
    ch_types = ['eeg']

    # Use magnetometer and gradiometer MEG channels:
    ch_types = ['mag', 'grad']

    # Currently does not work and will raise an error message:
    ch_types = ['meg', 'eeg']
    ```
"""

data_type: Optional[Literal['meg', 'eeg']] = None
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
    data_type = 'eeg'
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

eog_channels: Optional[Iterable[str]] = None
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

???+ example "Example"
    Treat `Fp1` as virtual EOG channel:
    ```python
    eog_channels = ['Fp1']
    ```

    Treat `Fp1` and `Fp2` as virtual EOG channels:
    ```python
    eog_channels = ['Fp1', 'Fp2']
    ```
"""

eeg_bipolar_channels: Optional[Dict[str, Tuple[str, str]]] = None
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

Note: Note
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

eeg_reference: Union[Literal['average'], str, Iterable['str']] = 'average'
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

eeg_template_montage: Optional[str] = None
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

drop_channels: Iterable[str] = []
"""
Names of channels to remove from the data. This can be useful, for example,
if you have added a new bipolar channel via `eeg_bipolar_channels` and now wish
to remove the anode, cathode, or both.

???+ example "Example"
    Exclude channels `Fp1` and `Cz` from processing:
    ```python
    drop_channels = ['Fp1', 'Cz]
    ```
"""

analyze_channels: Union[
    Literal['all'], Literal['ch_types'], Iterable['str']] = 'ch_types'
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

reader_extra_params: dict = {}
"""
Parameters to be passed to `read_raw_bids()` calls when importing raw data.

???+ example "Example"
    Enforce units for EDF files:
    ```python
    reader_extra_params = {"units": "uV"}
    ```
"""

###############################################################################
# BREAK DETECTION
# ---------------

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

min_break_duration: float = 15.
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

t_break_annot_start_after_previous_event: float = 5.
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

t_break_annot_stop_before_next_event: float = 5.
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

###############################################################################
# MAXWELL FILTER PARAMETERS
# -------------------------
# done in 01-import_and_maxfilter.py

find_flat_channels_meg: bool = False
"""
Auto-detect "flat" channels (i.e. those with unusually low variability) and
mark them as bad.
"""

find_noisy_channels_meg: bool = False
"""
Auto-detect "noisy" channels and mark them as bad.
"""

use_maxwell_filter: bool = False
"""
Whether or not to use Maxwell filtering to preprocess the data.

warning:
    If the data were recorded with internal active compensation (MaxShield),
    they need to be run through Maxwell filter to avoid distortions.
    Bad channels need to be set through BIDS channels.tsv and / or via the
    `find_flat_channels_meg` and `find_noisy_channels_meg` options above
    before applying Maxwell filter.
"""

mf_st_duration: Optional[float] = None
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

mf_head_origin: Union[Literal['auto'], ArrayLike] = 'auto'
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

mf_reference_run: Optional[str] = None
"""
Despite all possible care to avoid movements in the MEG, the participant
will likely slowly drift down from the Dewar or slightly shift the head
around in the course of the recording session. Hence, to take this into
account, we are realigning all data to a single position. For this, you need
to define a reference run (typically the one in the middle of
the recording session).

Which run to take as the reference for adjusting the head position of all
runs. If `None`, pick the first run.

???+ example "Example"
    ```python
    mf_reference_run = '01'  # Use run "01"
    ```
"""

mf_cal_fname: Optional[str] = None
"""
warning:
     This parameter should only be used for BIDS datasets that don't store
     the fine-calibration file
     [according to BIDS](https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#cross-talk-and-fine-calibration-files).
Path to the Maxwell Filter calibration file. If None the recommended
location is used.
???+ example "Example"
    ```python
    mf_cal_fname = '/path/to/your/file/calibration_cal.dat'
    ```
"""  # noqa : E501

mf_ctc_fname: Optional[str] = None
"""
Path to the Maxwell Filter cross-talk file. If None the recommended
location is used.
warning:
     This parameter should only be used for BIDS datasets that don't store
     the cross-talk file
     [according to BIDS](https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#cross-talk-and-fine-calibration-files).
???+ example "Example"
    ```python
    mf_ctc_fname = '/path/to/your/file/crosstalk_ct.fif'
    ```
"""  # noqa : E501

###############################################################################
# STIMULATION ARTIFACT
# --------------------
# used in 01-import_and_maxfilter.py

fix_stim_artifact: bool = False
"""
Apply interpolation to fix stimulation artifact.

???+ example "Example"
    ```python
    fix_stim_artifact = False
    ```
"""

stim_artifact_tmin: float = 0.
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

###############################################################################
# FREQUENCY FILTERING & RESAMPLING
# --------------------------------
# done in 02-frequency_filter.py

l_freq: Optional[float] = None
"""
The low-frequency cut-off in the highpass filtering step.
Keep it None if no highpass filtering should be applied.
"""

h_freq: Optional[float] = 40.
"""
The high-frequency cut-off in the lowpass filtering step.
Keep it None if no lowpass filtering should be applied.
"""

l_trans_bandwidth: Union[float, Literal['auto']] = 'auto'
"""
Specifies the transition bandwidth of the
highpass filter. By default it's `'auto'` and uses default MNE
parameters.
"""

h_trans_bandwidth: Union[float, Literal['auto']] = 'auto'
"""
Specifies the transition bandwidth of the
lowpass filter. By default it's `'auto'` and uses default MNE
parameters.
"""

raw_resample_sfreq: Optional[float] = None
"""
Specifies at which sampling frequency the data should be resampled.
If `None`, then no resampling will be done.

???+ example "Example"
    ```python
    raw_resample_sfreq = None  # no resampling
    raw_resample_sfreq = 500  # resample to 500Hz
    ```
"""

###############################################################################
# DECIMATION
# ----------

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


###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------

rename_events: dict = dict()
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

on_rename_missing_events: Literal['ignore', 'warn', 'raise'] = 'raise'
"""
How to handle the situation where you specified an event to be renamed via
`rename_events`, but this particular event is not present in the data. By
default, we will raise an exception to avoid accidental mistakes due to typos;
however, if you're sure what you're doing, you may change this to `'warn'`
to only get a warning instead, or `'ignore'` to ignore it completely.
"""

###############################################################################
# HANDLING OF REPEATED EVENTS
# ---------------------------

event_repeated: Literal['error', 'drop', 'merge'] = 'error'
"""
How to handle repeated events. We call events "repeated" if more than one event
occurred at the exact same time point. Currently, MNE-Python cannot handle
this situation gracefully when trying to create epochs, and will throw an
error. To only keep the event of that time point ("first" here referring to
the order that events appear in `*_events.tsv`), pass `'drop'`. You can also
request to create a new type of event by merging repeated events by setting
this to `'merge'`.

warning:
    The `'merge'` option is entirely untested in the MNE BIDS Pipeline as of
    April 1st, 2021.
"""

###############################################################################
# EPOCHING
# --------

epochs_metadata_tmin: Optional[float] = None
"""
The beginning of the time window for metadata generation, in seconds,
relative to the time-locked event of the respective epoch. This may be less
than or larger than the epoch's first time point. If `None`, use the first
time point of the epoch.
"""

epochs_metadata_tmax: Optional[float] = None
"""
Same as `epochs_metadata_tmin`, but specifying the **end** of the time
window for metadata generation.
"""

epochs_metadata_keep_first: Optional[Iterable[str]] = None
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

epochs_metadata_keep_last: Optional[Iterable[str]] = None
"""
Same as `epochs_metadata_keep_first`, but for keeping the **last**
occurrence of matching event types. The columns indicating the event types
will be named with a `last_` instead of a `first_` prefix.
"""

epochs_metadata_query: Optional[str] = None
"""
A [metadata query][https://mne.tools/stable/auto_tutorials/epochs/30_epochs_metadata.html]
specifying which epochs to keep. If the query fails because it refers to an
unknown metadata column, a warning will be emitted and all epochs will be kept.

???+ example "Example"
    Only keep epochs without a `response_missing` event:
    ```python
    epochs_metadata_query = ['response_missing.isna()']
    ```
"""  # noqa: E501

conditions: Optional[Union[Iterable[str], Dict[str, str]]] = None
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
[`task_is_rest`][mne_bids_pipeline._config.task_is_rest] is not `True`, we will raise an error.

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
"""  # noqa : E501

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

task_is_rest: bool = False
"""
Whether the task should be treated as resting-state data.
"""

rest_epochs_duration: Optional[float] = None
"""
Duration of epochs in seconds.
"""

rest_epochs_overlap: Optional[float] = None
"""
Overlap between epochs in seconds. This is used if the task is `'rest'`
and when the annotations do not contain any stimulation or behavior events.
"""

baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0)
"""
Specifies which time interval to use for baseline correction of epochs;
if `None`, no baseline correction is applied.

???+ example "Example"
    ```python
    baseline = (None, 0)  # beginning of epoch until time point zero
    ```
"""

contrasts: Iterable[
    Union[
        Tuple[str, str],
        ArbitraryContrast
    ]
] = []
"""
The conditions to contrast via a subtraction of ERPs / ERFs. The list elements
can either be tuples or dictionaries (or a mix of both). Each element in the
list corresponds to a single contrast.

A tuple specifies a one-vs-one contrast, where the second condition is
subtraced from the first.

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

report_evoked_n_time_points: Optional[int] = None
"""
Specifies the number of time points to display for each evoked
in the report. If None it defaults to the current default in MNE-Python.

???+ example "Example"
    Only display 5 time points per evoked
    ```python
    report_evoked_n_time_points = 5
    ```
"""

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run steps 5a and 6a
# if you choose SSP, run steps 5b and 6b
#
# Currently you cannot use both.

spatial_filter: Optional[Literal['ssp', 'ica']] = None
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
"""

min_ecg_epochs: int = 5
"""
Minimal number of ECG epochs needed to compute SSP or ICA rejection.
"""

min_eog_epochs: int = 5
"""
Minimal number of EOG epochs needed to compute SSP or ICA rejection.
"""


# Rejection based on SSP
# ~~~~~~~~~~~~~~~~~~~~~~


n_proj_eog: Dict[str, float] = dict(n_mag=1, n_grad=1, n_eeg=1)
"""
Number of SSP vectors to create for EOG artifacts for each channel type.
"""

n_proj_ecg: Dict[str, float] = dict(n_mag=1, n_grad=1, n_eeg=1)
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

ssp_meg: Literal['separate', 'combined', 'auto'] = 'auto'
"""
Whether to compute SSP vectors for MEG channels separately (`'separate'`)
or jointly (`'combined'`) for magnetometers and gradiomenters. When using
Maxwell filtering, magnetometer and gradiometer signals are synthesized from
multipole moments jointly and are no longer independent, so it can be useful to
estimate projectors from all MEG sensors simultaneously. The default is
`'auto'`, which will use `'combined'` when Maxwell filtering is used and
`'separate'` otherwise.
"""

ssp_reject_ecg: Optional[
    Union[
        Dict[str, float],
        Literal['autoreject_global']
    ]
] = None
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

ssp_reject_eog: Optional[
    Union[
        Dict[str, float],
        Literal['autoreject_global']
    ]
] = None
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


# Rejection based on ICA
# ~~~~~~~~~~~~~~~~~~~~~~


ica_reject: Optional[Dict[str, float]] = None
"""
Peak-to-peak amplitude limits to exclude epochs from ICA fitting.

This allows you to remove strong transient artifacts, which could negatively
affect ICA performance.

This will also be applied to ECG and EOG epochs created during preprocessing.

The BIDS Pipeline will automatically try to detect EOG and ECG artifacts in
your data, and remove them. For this to work properly, it is recommended
to **not** specify rejection thresholds for EOG and ECG channels here –
otherwise, ICA won't be able to "see" these artifacts.

If `None` (default), do not apply artifact rejection. If a dictionary,
manually specify peak-to-peak rejection thresholds (see examples).

???+ example "Example"
    ```python
    ica_reject = {'grad': 10e-10, 'mag': 20e-12, 'eeg': 400e-6}
    ica_reject = {'grad': 15e-10}
    ica_reject = None  # no rejection
    ```
"""

ica_algorithm: Literal['picard', 'fastica', 'extended_infomax'] = 'picard'
"""
The ICA algorithm to use.
"""

ica_l_freq: Optional[float] = 1.
"""
The cutoff frequency of the high-pass filter to apply before running ICA.
Using a relatively high cutoff like 1 Hz will remove slow drifts from the
data, yielding improved ICA results. Must be set to 1 Hz or above.

Set to `None` to not apply an additional high-pass filter.

Note: Note
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

ica_n_components: Optional[Union[float, int]] = 0.8
"""
MNE conducts ICA as a sort of a two-step procedure: First, a PCA is run
on the data (trying to exclude zero-valued components in rank-deficient
data); and in the second step, the principal components are passed
to the actual ICA. You can select how many of the total principal
components to pass to ICA – it can be all or just a subset. This determines
how many independent components to fit, and can be controlled via this
setting.

If int, specifies the number of principal components that are passed to the
ICA algorithm, which will be the number of independent components to
fit. It must not be greater than the rank of your data (which is typically
the number of channels, but may be less in some cases).

If float between 0 and 1, all principal components with cumulative
explained variance less than the value specified here will be passed to
ICA.

If `None`, **all** principal components will be used.

This setting may drastically alter the time required to compute ICA.
"""

ica_decim: Optional[int] = None
"""
The decimation parameter to compute ICA. If 5 it means
that 1 every 5 sample is used by ICA solver. The higher the faster
it is to run but the less data you have to compute a good ICA. Set to
`1` or `None` to not perform any decimation.
"""

ica_ctps_ecg_threshold: float = 0.1
"""
The threshold parameter passed to `find_bads_ecg` method.
"""

ica_eog_threshold: float = 3.0
"""
The threshold to use during automated EOG classification. Lower values mean
that more ICs will be identified as EOG-related. If too low, the
false-alarm rate increases dramatically.
"""


# Rejection based on peak-to-peak amplitude
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

reject: Optional[
    Union[Dict[str, float],
          Literal['autoreject_global']]
] = None
"""
Peak-to-peak amplitude limits to mark epochs as bad. This allows you to remove
epochs with strong transient artifacts.

If `None` (default), do not apply artifact rejection. If a dictionary,
manually specify rejection thresholds (see examples).  If
`'autoreject_global'`, use [`autoreject`](https://autoreject.github.io) to find
suitable "global" rejection thresholds for each channel type, i.e. `autoreject`
will generate a dictionary with (hopefully!) optimal thresholds for each
channel type.

The thresholds provided here must be at least as stringent as those in
[`ica_reject`][mne_bids_pipeline._config.ica_reject] if using ICA. In case of
`'autoreject_global'`, thresholds for any channel that do not meet this
requirement will be automatically replaced with those used in `ica_reject`.

Note: Note
      The rejection is performed **after** SSP or ICA, if any of those methods
      is used. To reject epochs **before** fitting ICA, see the
      [`ica_reject`][mne_bids_pipeline._config.ica_reject] setting.

If `None` (default), do not apply automated rejection. If a dictionary,
manually specify rejection thresholds (see examples).  If `'auto'`, use
[`autoreject`](https://autoreject.github.io) to find suitable "global"
rejection thresholds for each channel type, i.e. `autoreject` will generate
a dictionary with (hopefully!) optimal thresholds for each channel type. Note
that using `autoreject` can be a time-consuming process.

Note: Note
      `autoreject` basically offers two modes of operation: "global" and
      "local". In "global" mode, it will try to estimate one rejection
      threshold **per channel type.** In "local" mode, it will generate
      thresholds **for each individual channel.** Currently, the BIDS Pipeline
      only supports the "global" mode.

???+ example "Example"
    ```python
    reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
    reject = {'eeg': 100e-6, 'eog': 250e-6}
    reject = None  # no rejection based on PTP amplitude
    ```
"""

reject_tmin: Optional[float] = None
"""
Start of the time window used to reject epochs. If `None`, the window will
start with the first time point.
???+ example "Example"
    ```python
    reject_tmin = -0.1  # 100 ms before event onset.
    ```
"""

reject_tmax: Optional[float] = None
"""
End of the time window used to reject epochs. If `None`, the window will end
with the last time point.
???+ example "Example"
    ```python
    reject_tmax = 0.3  # 300 ms after event onset.
    ```
"""

###############################################################################
# DECODING
# --------

decode: bool = True
"""
Whether to perform decoding (MVPA) on the contrasts specified above as
[`contrasts`][mne_bids_pipeline._config.contrasts]. Classifiers will be trained
on entire epochs ("full-epochs decoding"), and separately on each time point
("time-by-time decoding"), trying to learn how to distinguish the contrasting
conditions.
"""

decoding_epochs_tmin: Optional[float] = 0.
"""
The first time sample to use for full epochs decoding. By default it starts
at 0. If None, it starts at the beginning of the epoch.
"""

decoding_epochs_tmax: Optional[float] = None
"""
The last time sample to use for full epochs decoding. By default it is set
to None so it ends at the end of the epoch.
"""

decoding_metric: str = 'roc_auc'
"""
The metric to use for estimating classification performance. It can be
`'roc_auc'` or `'accuracy'` – or any other metric supported by `scikit-learn`.

With ROC AUC, chance level is the same regardless of class balance, that is,
you don't need to be worried about **exactly** balancing class sizes.
"""

decoding_n_splits: int = 5
"""
The number of folds (also called "splits") to use in the cross-validation
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
"""  # noqa: E501

decoding_time_generalization_decim: int = 1
"""
Says how much to decimate data before time generalization decoding.
This is done in addition to the decimation done at the epochs level via the
[`epochs_decim`][mne_bids_pipeline._config.epochs_decim] parameter. This can be
used to greatly speed up time generalization at the cost of lower time
resolution in the resulting matrix.
"""

n_boot: int = 5000
"""
The number of bootstrap resamples when estimating the standard error and
confidence interval of the mean decoding scores.
"""

cluster_forming_t_threshold: Optional[float] = None
"""
The t-value threshold to use for forming clusters in the cluster-based
permutation test run on the the time-by-time decoding scores.
Data points with absolute t-values greater than this value
will be used to form clusters. If `None`, the threshold will be automatically
determined to correspond to a p-value of 0.05 for the given number of
participants in a one-tailed test.

Note: Note
    Only points with the same sign will be clustered together.
"""

cluster_n_permutations: int = 10_000
"""
The maximum number of permutations to perform in a cluster-based permutation
test to determine the significance of the decoding scores across participants.
"""

cluster_permutation_p_threshold: float = 0.05
"""
The alpha level (p-value, p threshold) to use for rejecting the null hypothesis
that the clusters show no significant difference between conditions. This is
used in the permutation test which takes place after forming the clusters.

Note: Note
    To control how clusters are formed, see
    [`cluster_forming_t_threshold`][mne_bids_pipeline._config.cluster_forming_t_threshold].
"""

###############################################################################
# GROUP AVERAGE SENSORS
# ---------------------

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

###############################################################################
# TIME-FREQUENCY
# --------------

time_frequency_conditions: Iterable[str] = []
"""
The conditions to compute time-frequency decomposition on.

???+ example "Example"
    ```python
    time_frequency_conditions = ['left', 'right']
    ```
"""

time_frequency_freq_min: Optional[float] = 8
"""
Minimum frequency for the time frequency analysis, in Hz.
???+ example "Example"
    ```python
    time_frequency_freq_min = 0.3  # 0.3 Hz
    ```
"""

time_frequency_freq_max: Optional[float] = 40
"""
Maximum frequency for the time frequency analysis, in Hz.
???+ example "Example"
    ```python
    time_frequency_freq_max = 22.3  # 22.3 Hz
    ```
"""

time_frequency_cycles: Optional[Union[float, ArrayLike]] = None
"""
The number of cycles to use in the Morlet wavelet. This can be a single number
or one per frequency, where frequencies are calculated via
`np.arange(time_frequency_freq_min, time_frequency_freq_max)`.
If `None`, uses
`np.arange(time_frequency_freq_min, time_frequency_freq_max) / 3`.
"""

time_frequency_subtract_evoked: bool = False
"""
Whether to subtract the evoked signal (averaged across all epochs) from the
epochs before passing them to time-frequency analysis. Set this to `True` to
highlight induced activity.

Note: Note
     This also applies to CSP analysis.
"""

###############################################################################
# TIME-FREQUENCY CSP
# ------------------

decoding_csp: bool = False
"""
Whether to run decoding via Common Spatial Patterns (CSP) analysis on the
data. CSP takes as input data covariances that are estimated on different
time and frequency ranges. This allows to obtain decoding scores defined over
time and frequency.
"""

decoding_csp_times: Optional[ArrayLike] = np.linspace(
    max(0, epochs_tmin),
    epochs_tmax,
    num=6
)
"""
The edges of the time bins to use for CSP decoding.
Must contain at least two elements. By default, 5 equally-spaced bins are
created across the non-negative time range of the epochs.
All specified time points must be contained in the epochs interval.
If `None`, do not perform **time-frequency** analysis, and only run CSP on
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

decoding_csp_freqs: Dict[str, ArrayLike] = {
    'custom': [
        time_frequency_freq_min,
        (time_frequency_freq_max + time_frequency_freq_min) / 2,  # noqa: E501
        time_frequency_freq_max
    ]
}
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

time_frequency_baseline: Optional[Tuple[float, float]] = None
"""
Baseline period to use for the time-frequency analysis. If `None`, no baseline.
???+ example "Example"
    ```python
    time_frequency_baseline = (None, 0)
    ```
"""

time_frequency_baseline_mode: str = 'mean'
"""
Baseline mode to use for the time-frequency analysis. Can be chosen among:
"mean" or "ratio" or "logratio" or "percent" or "zscore" or "zlogratio".
???+ example "Example"
    ```python
    time_frequency_baseline_mode = 'mean'
    ```
"""

time_frequency_crop: Optional[dict] = None
"""
Period and frequency range to crop the time-frequency analysis to.
If `None`, no cropping.

???+ example "Example"
    ```python
    time_frequency_crop = dict(tmin=-0.3, tmax=0.5, fmin=5, fmax=20)
    ```
"""

###############################################################################
# SOURCE ESTIMATION PARAMETERS
# ----------------------------
#

run_source_estimation: bool = True
"""
Whether to run source estimation processing steps if not explicitly requested.
"""

use_template_mri: Optional[str] = None
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

bem_mri_images: Literal['FLASH', 'T1', 'auto'] = 'auto'
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

mri_t1_path_generator: Optional[
    Callable[[BIDSPath], BIDSPath]
] = None
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

Note: Note
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

mri_landmarks_kind: Optional[
    Callable[[BIDSPath], str]
] = None
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

spacing: Union[Literal['oct5', 'oct6', 'ico4', 'ico5', 'all'], int] = 'oct6'
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

loose: Union[float, Literal['auto']] = 0.2
"""
Value that weights the source variances of the dipole components
that are parallel (tangential) to the cortical surface. If `0`, then the
inverse solution is computed with **fixed orientation.**
If `1`, it corresponds to **free orientation.**
The default value, `'auto'`, is set to `0.2` for surface-oriented source
spaces, and to `1.0` for volumetric, discrete, or mixed source spaces,
unless `fixed is True` in which case the value 0. is used.
"""

depth: Optional[Union[float, dict]] = 0.8
"""
If float (default 0.8), it acts as the depth weighting exponent (`exp`)
to use (must be between 0 and 1). None is equivalent to 0, meaning no
depth weighting is performed. Can also be a `dict` containing additional
keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
(see docstring for details and defaults).
"""

inverse_method: Literal['MNE', 'dSPM', 'sLORETA', 'eLORETA'] = 'dSPM'
"""
Use minimum norm, dSPM (default), sLORETA, or eLORETA to calculate the inverse
solution.
"""

noise_cov: Union[
    Tuple[Optional[float], Optional[float]],
    Literal['emptyroom', 'rest', 'ad-hoc'],
    Callable[[BIDSPath], mne.Covariance]
] = (None, 0)
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
the generated evoked data.

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
        bp = bids_path.copy().update(task='rest', run=None, suffix='meg')
        raw_rest = mne_bids.read_raw_bids(bp)
        raw.crop(tmin=5, tmax=60)
        cov = mne.compute_raw_covariance(raw, rank='info')
        return cov
    ```
"""

source_info_path_update: Optional[Dict[str, str]] = dict(suffix='ave')
"""
When computing the forward and inverse solutions, by default the pipeline
retrieves the `mne.Info` object from the cleaned evoked data. However, in
certain situations you may wish to use a different `Info`.

This parameter allows you to explicitly specify from which file to retrieve the
`mne.Info` object. Use this parameter to supply a dictionary to
`BIDSPath.update()` during the forward and inverse processing steps.

???+ example "Example"
    Use the `Info` object stored in the cleaned epochs:
    ```python
    source_info_path_update = {'processing': 'clean',
                               'suffix': 'epo'}
    ```
"""

inverse_targets: List[Literal['evoked']] = ['evoked']
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

report_stc_n_time_points: Optional[int] = None
"""
Specifies the number of time points to display for each source estimates
in the report. If None it defaults to the current default in MNE-Python.

???+ example "Example"
    Only display 5 images per source estimate:
    ```python
    report_stc_n_time_points = 5
    ```
"""

###############################################################################
# Execution
# ---------

N_JOBS: int = 1
"""
Specifies how many subjects you want to process in parallel. If `1`, disables
parallel processing.
"""

parallel_backend: Literal['loky', 'dask'] = 'loky'
"""
Specifies which backend to use for parallel job execution. `loky` is the
default backend used by `joblib`. `dask` requires [`Dask`](https://dask.org) to
be installed. Ignored if [`N_JOBS`][mne_bids_pipeline._config.N_JOBS] is set to
`1`.
"""

dask_open_dashboard: bool = False
"""
Whether to open the Dask dashboard in the default webbrowser automatically.
Ignored if `parallel_backend` is not `'dask'`.
"""

dask_temp_dir: Optional[PathLike] = None
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

dask_worker_memory_limit: str = '10G'
"""
The maximum amount of RAM per Dask worker.
"""

random_state: Optional[int] = 42
"""
You can specify the seed of the random number generator (RNG).
This setting is passed to the ICA algorithm and to the decoding function,
ensuring reproducible results. Set to `None` to avoid setting the RNG
to a defined state.
"""

shortest_event: int = 1
"""
Minimum number of samples an event must last. If the
duration is less than this, an exception will be raised.
"""

log_level: Literal['info', 'error'] = 'info'
"""
Set the pipeline logging verbosity.
"""

mne_log_level: Literal['info', 'error'] = 'error'
"""
Set the MNE-Python logging verbosity.
"""

on_error: Literal['continue', 'abort', 'debug'] = 'abort'
"""
Whether to abort processing as soon as an error occurs, continue with all other
processing steps for as long as possible, or drop you into a debugger in case
of an error.

Note: Note
    Enabling debug mode deactivates parallel processing.
"""

memory_location: Optional[Union[PathLike, bool]] = True
"""
If not None (or False), caching will be enabled and the cache files will be
stored in the given directory. The default (True) will use a
`'joblib'` subdirectory in the BIDS derivative root of the dataset.
"""

memory_file_method: Literal['mtime', 'hash'] = 'mtime'
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

config_validation: Literal['raise', 'warn', 'ignore'] = 'raise'
"""
How strictly to validate the configuration. Errors are always raised for
invalid entries (e.g., not providing `ch_types`). This setting controls
how to handle *possibly* or *likely* incorrect entries, such as likely
misspellings (e.g., providing `session` instead of `sessions`).
"""
