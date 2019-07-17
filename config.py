"""Set the configuration parameters for the study.

You need to define an environment variable `BIDS_ROOT` to point to the root
of your BIDS dataset to be analyzed.

"""
import importlib
import os
from collections import defaultdict

from bids import BIDSLayout
import numpy as np


# Get the bids_root from an environment variable, raise an error if not found
bids_root = os.getenv('BIDS_ROOT', False)
if not bids_root:
    raise RuntimeError('You need to define an environment variable '
                       '`BIDS_ROOT` pointing to the root of your BIDS dataset')

# Make a pybids layout to easily get subjects, tasks, and other BIDS params
layout = BIDSLayout(bids_root)

# ``subjects_dir`` : str
#   Path to the directory that contains the MRI data files and their
#   derivativesfor all subjects. Specifically, the ``subjects_dir`` is the
#   $SUBJECTS_DIR used by the Freesurfer software.

subjects_dir = os.path.join(bids_root, 'derivatives', 'freesurfer', 'subjects')


# ``plot``  : boolean
#   If True, the scripts will generate plots.
#   If running the scripts from a notebook or spyder
#   run %matplotlib qt in the command line to get the plots in extra windows

plot = True
plot = False


# BIDS params
# see: bids-specification.rtfd.io/en/latest/99-appendices/04-entity-table.html
ses = layout.get(return_type='id', target='session')
ses = ses if ses else None

# XXX: take only first task for now
task = layout.get(return_type='id', target='task')[0]

run = layout.get(return_type='id', target='run')
run = run if run else None

acq = layout.get(return_type='id', target='acquisition')
acq = acq if acq else None

proc = layout.get(return_type='id', target='proc')
proc = proc if proc else None

rec = layout.get(return_type='id', target='reconstruction')
rec = rec if rec else None

space = None

kinds = layout.get(return_type='id', target='datatype')
if 'eeg' in kinds and 'meg' in kinds:
    raise RuntimeError('Found data of kind EEG and MEG. Please specify an '
                       'environment variable `BIDS_KIND` and set it to the '
                       'kind you want to analyze.')
    kind = kinds[0]
elif 'eeg' in kinds:
    kind = 'eeg'
elif 'meg' in kinds:
    kind = 'meg'
else:
    raise ValueError('The study template is intended for EEG or MEG data, but'
                     'your dataset does not contain data of either type.')

# ``subjects_list`` : list of str
#   To define the list of participants, we use a list with all the anonymized
#   participant names. Even if you plan on analyzing a single participant, it
#   needs to be set up as a list with a single element, as in the 'example'
#   subjects_list = ['SB01']

# subjects_list = ['05', '06', '07']

subjects_list = layout.get(return_type='id', target='subject')

# subjects_list = ['05']

# ``exclude_subjects`` : list of str
#   Now you can specify subjects to exclude from the group study:
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)

exclude_subjects = []

# ``ch_types``  : list of st
#    The list of channel types to consider.
#
# Example
# ~~~~~~~
# >>> ch_types = ['meg', 'eeg']  # to use MEG and EEG channels
# or
# >>> ch_types = ['meg']  # to use only MEG
# or
# >>> ch_types = ['grad']  # to use only gradiometer MEG channels

# ch_types = ['eeg']
ch_types = ['meg']




###############################################################################
# BAD CHANNELS
# ------------
# needed for 01-import_and_filter.py

# ``bads`` : dict of list | dict of dict
#    Bad channels are noisy sensors that *must* to be listed
#    *before* maxfilter is applied. You can use the dict of list structure
#    of you have bad channels that are the same for all runs.
#    Use the dict(dict) if you have many runs or if noisy sensors are changing
#    across runs.
#
# Example
# ~~~~~~~
# >>> bads = defaultdict(list)
# >>> bads['sample'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
# or
# >>> def default_bads():
# >>>     return dict(run01=[], run02=[])
# >>>
# >>> bads = defaultdict(default_bads)
# >>> bads['subject01'] = dict(run01=['MEG1723', 'MEG1722'], run02=['MEG1723'])
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# During the acquisition of your MEG / EEG data, systematically list and keep
# track of the noisy sensors. Here, put the number of runs you ideally expect
# to have per participant. Use the simple dict if you don't have runs or if
# the same sensors are noisy across all runs.

# # XXX bads should in the BIDS files
# bads = defaultdict(list)
# bads['sample'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

###############################################################################
# DEFINE ADDITIONAL CHANNELS
# --------------------------
# needed for 01-import_and_filter.py

# ``rename_channels`` : dict rename channels
#    Here you name or replace extra channels that were recorded, for instance
#    EOG, ECG.
#
# Example
# ~~~~~~~
# Here rename EEG061 to EOG061, EEG062 to EOG062, EEG063 to ECG063:
# >>> rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062',
#                        'EEG063': 'ECG063'}

# XXX should be done automatically from BIDS ?
rename_channels = None

# ``set_channel_types``: dict
#   Here you define types of channels to pick later.
#
# Example
# ~~~~~~~
# >>> set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog',
#                          'EEG063': 'ecg', 'EEG064': 'misc'}

# XXX should not be necessary
set_channel_types = None

###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 01-import_and_filter.py

# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
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
# coult be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! :)


# ``l_freq`` : float
#   The low-frequency cut-off in the highpass filtering step.
#   Keep it None if no highpass filtering should be applied.

l_freq = 1.

# ``h_freq`` : float
#   The high-frequency cut-off in the lowpass filtering step.
#   Keep it None if no lowpass filtering should be applied.

h_freq = 40.

###############################################################################
# MAXFILTER PARAMETERS
# --------------------
#
# ``use_maxwell_filter`` : bool
#   Use or not maxwell filter to preprocess the data.

use_maxwell_filter = False

# There are two kinds of maxfiltering: SSS and tSSS
# [SSS = signal space separation ; tSSS = temporal signal space separation]
# (Taulu et al, 2004): http://cds.cern.ch/record/709081/files/0401166.pdf
#
# ``mf_st_duration`` : float | None
#    If not None, apply spatiotemporal SSS (tSSS) with specified buffer
#    duration (in seconds). MaxFilter™'s default is 10.0 seconds in v2.2.
#    Spatiotemporal SSS acts as implicitly as a high-pass filter where the
#    cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
#    buffers are generally better as long as your system can handle the
#    higher memory usage. To ensure that each window is processed
#    identically, choose a buffer length that divides evenly into your data.
#    Any data at the trailing edge that doesn't fit evenly into a whole
#    buffer window will be lumped into the previous buffer.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you are interested in low frequency activity (<0.1Hz), avoid using tSSS
# and set mf_st_duration to None
#
# If you are interested in low frequency above 0.1 Hz, you can use the
# default mf_st_duration to 10 s meaning it acts like a 0.1 Hz highpass filter.
#
# Example
# ~~~~~~~
# >>> mf_st_duration = None
# or
# >>> mf_st_duration = 10.  # to apply tSSS with 0.1Hz highpass filter.

mf_st_duration = None

# ``mf_head_origin`` : array-like, shape (3,) | 'auto'
#   Origin of internal and external multipolar moment space in meters.
#   If 'auto', it will be estimated from headshape points.
#   If automatic fitting fails (e.g., due to having too few digitization
#   points), consider separately calling the fitting function with different
#   options or specifying the origin manually.
#
# Example
# ~~~~~~~
# >>> mf_head_origin = 'auto'

mf_head_origin = 'auto'

# ``cross talk`` : str
#   Path to the cross talk file
#
#
# ``calibration`` : str
#   Path to the calibration file.
#
#
# These 2 files should be downloaded and made available for running
# maxwell filtering.
#
# Example
# ~~~~~~~
# >>> cal_files_path = os.path.join(study_path, 'SSS')
# >>> mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_mgh.fif')
# >>> mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_mgh.dat')
#
# Warning
# ~~~~~~~
# These 2 files are site and machine specific files that provide information
# about the environmental noise. For practical purposes, place them in your
# study folder.
#
# At NeuroSpin: ct_sparse and sss_call are on the meg_tmp server

# cal_files_path = os.path.join(study_path, 'SSS')
# mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_mgh.fif')
# mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_mgh.dat')

# Despite all possible care to avoid movements in the MEG, the participant
# will likely slowly drift down from the Dewar or slightly shift the head
# around in the course of the recording session. Hence, to take this into
# account, we are realigning all data to a single position. For this, you need
# to define a reference run (typically the one in the middle of
# the recording session).
#
# ``mf_reference_run``  : int
#   Which run to take as the reference for adjusting the head position of all
#   runs.
#
# Example
# ~~~~~~~
# >>> mf_reference_run = 0  # to use the first run

mf_reference_run = 0

###############################################################################
# RESAMPLING
# ----------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data
#
# ``resample_sfreq``  : float
#   Specifies at which sampling frequency the data should be resampled.
#   If None then no resampling will be done.
#
# Example
# ~~~~~~~
# >>> resample_sfreq = None  # no resampling
# or
# >>> resample_sfreq = 500  # resample to 500Hz

resample_sfreq = None

# ``decim`` : int
#   Says how much to decimate data at the epochs level.
#   It is typically an alternative to the `resample_sfreq` parameter that
#   can be used for resampling raw data. 1 means no decimation.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Decimation requires to lowpass filtered the data to avoid aliasing.
# Note that using decimation is much faster than resampling.
#
# Example
# ~~~~~~~
# >>> decim = 1  # no decimation
# or
# >>> decim = 4  # decimate by 4 ie devide sampling frequency by 4

decim = 1

###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Have a look at your raw data and train yourself to detect a blink, a heart
# beat and an eye movement.
# You can do a quick average of blink data and check what the amplitude looks
# like.
#
#  ``reject`` : dict | None
#    The rejection limits to make some epochs as bads.
#    This allows to remove strong transient artifacts.
#    If you want to reject and retrieve blinks later, e.g. with ICA,
#    don't specify a value for the eog channel (see examples below).
#    Make sure to include values for eeg if you have EEG data
#
# Note
# ~~~~
# These numbers tend to vary between subjects.. You might want to consider
# using the autoreject method by Jas et al. 2018.
# See https://autoreject.github.io
#
# Example
# ~~~~~~~
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 200e-6}
# >>> reject = None

reject = {'grad': 4000e-13, 'mag': 4e-12}
# reject = {'eeg': 150e-6}

###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float
#    A float in seconds that gives the start time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmin = -0.2  # take 200ms before event onset.

tmin = -0.2

# ``tmax``: float
#    A float in seconds that gives the end time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmax = 0.5  # take 500ms after event onset.

tmax = 0.5

# ``trigger_time_shift`` : float | None
#    If float it specifies the offset for the trigger and the stimulus
#    (in seconds). You need to measure this value for your specific
#    experiment/setup.
#
# Example
# ~~~~~~~
# >>> trigger_time_shift = 0  # don't apply any offset

trigger_time_shift = 0.

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.
#
# Example
# ~~~~~~~
# >>> baseline = (None, 0)  # baseline between tmin and 0

baseline = (None, 0)

# ``stim_channel`` : str
#    The name of the stimulus channel, which contains the events.
#
# Example
# ~~~~~~~
# >>> stim_channel = 'STI 014'  # or 'STI101'

# XXX not needed if bids events are present
# stim_channel = 'STI 014'

# ``min_event_duration`` : float
#    The minimal duration of the events you want to extract (in seconds).
#
# Example
# ~~~~~~~
# >>> min_event_duration = 0.002  # 2 miliseconds

# XXX not needed if bids events are present
# min_event_duration = 0.002

#  `event_id`` : dict
#    Dictionary that maps events (trigger/marker values)
#    to conditions.
#
# Example
# ~~~~~~~
# >>> event_id = {'auditory/left': 1, 'auditory/right': 2}`
# or
# >>> event_id = {'Onset': 4} with conditions = ['Onset']

# event_id = {'auditory/left': 1, 'auditory/right': 2}

#  `conditions`` : dict
#    List of condition names to consider. Must match the keys
#    in event_id.
#
# Example
# ~~~~~~~
# >>> conditions = ['auditory', 'visual']
# or
# >>> conditions = ['left', 'right']

# conditions = ['left', 'right']
conditions = ['somato_event1']

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run scripts 5a and 6a
# if you choose SSP, run scripts 5b and 6b
#
# Currently you cannot use both.
#
# ``use_ssp`` : bool
#    If True ICA should be used or not.

use_ssp = True

# ``use_ica`` : bool
#    If True ICA should be used or not.

use_ica = False

# ``ica_decim`` : int
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA.

ica_decim = 11


# ``default_reject_comps_factory`` : callable
#    A factory function that returns a default rejection component dictionary:
#    A dictionary that specifies the indices of the ICA components to reject
#    for each subject. For example you can use:
#    rejcomps_man['subject01'] = dict(eeg=[12], meg=[7])

def default_reject_comps_factory():
    """Return the default rejection component dictionary."""
    return dict(meg=[], eeg=[])


rejcomps_man = defaultdict(default_reject_comps_factory)

# ``ica_ctps_ecg_threshold``: float
#    The threshold parameter passed to `find_bads_ecg` method.

ica_ctps_ecg_threshold = 0.1

###############################################################################
# DECODING
# --------
#
# ``decoding_conditions`` : list
#    List of conditions to be classified.
#
# Example
# ~~~~~~~
# >>> decoding_conditions = []  # don't do decoding
# or
# >>> decoding_conditions = [('auditory', 'visual'), ('left', 'right')]

decoding_conditions = []
# decoding_conditions = [('left', 'right')]


# ``decoding_metric`` : str
#    The metric to use for cross-validation. It can be 'roc_auc' or 'accuracy'
#    or any metric supported by scikit-learn.

decoding_metric = 'roc_auc'

# ``decoding_n_splits`` : int
#    The number of folds (a.k.a. splits) to use in the cross-validation.

decoding_n_splits = 5

###############################################################################
# TIME-FREQUENCY
# --------------
#
# ``time_frequency_conditions`` : list
#    The conditions to compute time-frequency decomposition on.

# time_frequency_conditions = ['left', 'right']
time_frequency_conditions = []

###############################################################################
# SOURCE SPACE PARAMETERS
# -----------------------
#

# ``spacing`` : str
#    The spacing to use. Can be ``'ico#'`` for a recursively subdivided
#    icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
#    ``'all'`` for all points, or an integer to use appoximate
#    distance-based spacing (in mm).

spacing = 'oct6'

# ``mindist`` : float
#    Exclude points closer than this distance (mm) to the bounding surface.

mindist = 5

# ``loose`` : float in [0, 1] | 'auto'
#    Value that weights the source variances of the dipole components
#    that are parallel (tangential) to the cortical surface. If loose
#    is 0 then the solution is computed with fixed orientation,
#    and fixed must be True or "auto".
#    If loose is 1, it corresponds to free orientations.
#    The default value ('auto') is set to 0.2 for surface-oriented source
#    space and set to 1.0 for volumetric, discrete, or mixed source spaces,
#    unless ``fixed is True`` in which case the value 0. is used.

loose = 0.2

# ``depth`` : None | float | dict
#    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
#    to use (must be between 0 and 1). None is equivalent to 0, meaning no
#    depth weighting is performed. Can also be a `dict` containing additional
#    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
#    (see docstring for details and defaults).

depth = 0.8

# method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
#    Use minimum norm, dSPM (default), sLORETA, or eLORETA.

method = 'dSPM'

# smooth : int | None
#    Number of iterations for the smoothing of the surface data.
#    If None, smooth is automatically defined to fill the surface
#    with non-zero values. The default is spacing=None.

smooth = 10

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

# if not os.path.isdir(study_path):
#     os.mkdir(study_path)

# if not os.path.isdir(subjects_dir):
#    os.mkdir(subjects_dir)

###############################################################################
# ADVANCED
# --------
#
# ``l_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    highpass filter. By default it's `'auto'` and uses default mne
#    parameters.

l_trans_bandwidth = 'auto'

#  ``h_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    lowpass filter. By default it's `'auto'` and uses default mne
#    parameters.

h_trans_bandwidth = 'auto'

#  ``N_JOBS`` : int
#    An integer that specifies how many subjects you want to run in parallel.

N_JOBS = 1

# ``random_state`` : None | int | np.random.RandomState
#    To specify the random generator state. This allows to have
#    the results more reproducible between machines and systems.
#    Some methods like ICA need random values for initialisation.

random_state = 42

# ``shortest_event`` : int
#    Minimum number of samples an event must last. If the
#    duration is less than this an exception will be raised.

shortest_event = 1

# ``allow_maxshield``  : bool
#    To import data that was recorded with Maxshield on before running
#    maxfilter set this to True.

allow_maxshield = True

###############################################################################
# Overwrite with custom config options
#
# This works by setting an environment variable "MNE_BIDS_STUDY_CONFIG" to
# a string of the format "config_<dataset_name>.py", where <dataset_name>
# could be e.g., matchingpennies, or mne_sample_data

if "MNE_BIDS_STUDY_CONFIG" in os.environ:
    custom_cfg = importlib.import_module(os.environ['MNE_BIDS_STUDY_CONFIG'])
    for val in dir(custom_cfg):
        if not val.startswith('__'):
            exec("%s = custom_cfg.%s" % (val, val))
            print('Overwriting: %s' % val)

###############################################################################
# CHECKS
# --------
#
# --- --- You should not touch the next lines --- ---

if (use_maxwell_filter and
        len(set(ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
    raise ValueError('Cannot use maxwell filter without MEG channels.')

if use_ssp and use_ica:
    raise ValueError('Cannot use both SSP and ICA.')
