"""
===========
Config file
===========

Configuration parameters for the study. This should be in a folder called
``library/`` inside the ``processing/`` directory.
"""

import os
from collections import defaultdict
import numpy as np


# ``plot``  : boolean
#   If True, the scripts will generate plots.
#   If running the scripts from a notebook or spyder
#   run %matplotlib qt in the command line to get the plots in extra windows

plot = False


###############################################################################
# DIRECTORIES
# -----------
#
# ``study_path`` : str
#    Set the `study path`` where the data is stored on your system.
#
# Example
# ~~~~~~~
# >>> study_path = '../MNE-sample-data/'
# or
# >>> study_path = '/Users/sophie/repos/ExampleData/'

study_path = 'data/'

# ``subjects_dir`` : str
#   The ``subjects_dir`` contains the MRI files for all subjects.

subjects_dir = os.path.join(study_path, 'subjects')

# ``meg_dir`` : str
#   The ``meg_dir`` contains the MEG data in subfolders
#   named my_study_path/MEG/my_subject/

meg_dir = os.path.join(study_path, 'MEG')


###############################################################################
# SUBJECTS / RUNS
# ---------------
#
# ``study_name`` : str
#   This is the name of your experiment.
study_name = 'Localizer'

# ``subjects_list`` : list of str
#   To define the list of participants, we use a list with all the anonymized
#   participant names. Even if you plan on analyzing a single participant, it
#   needs to be set up as a list with a single element, as in the 'example'
#   subjects_list = ['SB01']

# To use all subjects use
subjects_list = ['SB01', 'SB02', 'SB04', 'SB05', 'SB06', 'SB07',
                 'SB08', 'SB09', 'SB10', 'SB11', 'SB12']
# else for speed and fast test you can use:

subjects_list = ['SB01']

# ``exclude_subjects`` : list of str
#   Now you can specify subjects to exclude from the group study:
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)

exclude_subjects = []

# ``runs`` : list of str
#   Define the names of your ``runs``
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# The naming should be consistent across participants. List the number of runs
# you ideally expect to have per participant. The scripts will issue a warning
# if there are less runs than is expected. If there is only just one file,
# leave empty!

runs = ['']  # ['run01', 'run02']

# ``eeg``  : bool
#    If true use the EEG channels

eeg = False  # True

# ``base_fname`` : str
#    This automatically generates the name for all files
#    with the variables specified above.
#    Normally you should not have to touch this

base_fname = '{subject}_' + study_name + '{extension}.fif'


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
#
# >>> def default_bads():
# >>>     return dict(run01=[], run02=[])
# >>>
# >>> bads = defaultdict(default_bads)
#
#   and to populate this, do:
#
# >>> bads['subject01'] = dict(run01=[12], run02=[7])
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# During the acquisition of your MEG / EEG data, systematically list and keep
# track of the noisy sensors. Here, put the number of runs you ideally expect
# to have per participant. Use the simple dict if you don't have runs or if
# the same sensors are noisy across all runs.

bads = defaultdict(list)
bads['SB01'] = ['MEG1723', 'MEG1722']
bads['SB04'] = ['MEG0543', 'MEG2333']
bads['SB06'] = ['MEG2632', 'MEG2033']


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
# >>> rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062',
#                        'EEG063': 'ECG063'}

rename_channels = None

# ``set_channel_types``: dict
#   Here you defines types of channels to pick later.
#
# Example
# ~~~~~~~
# >>> set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog',
#                          'EEG063': 'ecg', 'EEG064': 'misc'}

set_channel_types = None

###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 01-import_and_filter.py

# [Good Practice / Advice]
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border effects
#
# If you use this pipeline for evoked responses, a default filtering would be
# a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 40 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band
#
# If you use this pipeline for time-frequency analysis, a default filtering
# would be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band
#
# If you use are interested in the lowest frequencies, do not use a high-pass
# filter cut-off of l_freq = None
# If you need more fancy analysis, you are already likely past this kind
# of tips! :)


# ``l_freq`` : the low-frequency cut-off in the highpass filtering step.
#   Keep it None if no highpass filtering should be applied.

l_freq = 1.

# ``h_freq`` : the high-frequency cut-off in the lowpass filtering step.
#   Keep it None if no lowpass filtering should be applied.

h_freq = 40.

###############################################################################
# MAXFILTER PARAMETERS
# --------------------
#

# Download the ``cross talk`` and ``calibration`` files. Warning: these are
# site and machine specific files that provide information about the
# environmental noise.
# For practical purposes, place them in your study folder.
# At NeuroSpin: ct_sparse and sss_call are on the meg_tmp server

cal_files_path = os.path.join(study_path, 'system_calibration_files')
mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_nspn.fif')
mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_nspn.dat')

# [Good Practice / Advice]
# Despite all possible care to avoid movements in the MEG, the participant
# will likely slowly drift down from the Dewar or slightly shift the head
# around in the course of the recording session. Hence, to take this into
# account, we are realigning all data to a single position. For this, you need
# to define a reference run (typically the one in the middle of
# the recording session).

# ``mf_reference_run``  : integer
#   Which run to take as the reference for adjusting the head position of all
#   runs.

mf_reference_run = 0

# Set the origin for the head position

mf_head_origin = 'auto'

# [Good Practice / Advice]
# There are two kinds of maxfiltering: sss and tsss
# [sss = signal space separation ; tsss = temporal signal space separation]
# (Taulu et al, 2004): http://cds.cern.ch/record/709081/files/0401166.pdf
# If you are interested in low frequency activity (<0.1Hz), avoid using tsss
# and set mf_st_duration = None
# If you are interested in low frequency above 0.1 Hz, you can use the
# default mf_st_duration = 10 s
# Elekta default = 10s, meaning it acts like a 0.1 Hz highpass filter
# ``mf_st_duration `` : if None, no temporal-spatial filtering is applied
# during MaxFilter, otherwise, put a float that speficifies the buffer
# duration in seconds

mf_st_duration = None

###############################################################################
# RESAMPLING
# ----------
#
# [Good Practice / Advice]
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data
#
# ``resample_sfreq``  : a float that specifies at which sampling frequency
# the data should be resampled. If None then no resampling will be done.

resample_sfreq = 500.  # None

# ``decim`` : integer that says how much to decimate data at the epochs level.
# It is typically an alternative to the `resample_sfreq` parameter that
# can be used for resampling raw data. 1 means no decimation.

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

###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float
#    A float in seconds that gives the start time before event of an epoch.

tmin = -0.6

# ``tmax``: float
#    A float in seconds that gives the end time before event of an epoch.

tmax = 1.5

# ``trigger_time_shift`` : float | None
#    If float it specifies the offset for the trigger and the stimulus
#    (in seconds). You need to measure this value for your specific
#    experiment/setup.

trigger_time_shift = -0.0416

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.

baseline = (-.6, -.1)  # (None, 0.)

# ``stim_channel`` : str
#    The name of the stimulus channel, which contains the events.

stim_channel = 'STI101'  # 'STI014'# None

# ``min_event_duration`` : float
#    The minimal duration of the events you want to extract (in seconds).

min_event_duration = 0.002

#  `event_id`` : dict
#    Dictionary that maps events (trigger/marker values)
#    to conditions.
#
# Example
# ~~~~~~~
# >>> event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}`
# or
# >>> event_id = {'Onset': 4} with conditions = ['Onset']

event_id = {'incoherent/1': 33, 'incoherent/2': 35,
            'coherent/down': 37, 'coherent/up': 39}
conditions = ['incoherent', 'coherent']

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run scripts 5a and 6a
# if you choose SSP, run scripts 5b and 6b
# if you running both, your cleaned epochs will be the ones cleaned with the
# methods you run last (they overwrite each other)
#
#
# ``runica`` : bool
#    If True ICA should be used or not.

runica = True

# ``ica_decim`` : int
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA.

ica_decim = 11


# ``default_reject_comps`` : dict
#    A dictionary that specifies the indices of the ICA components to reject
#    for each subject. For example you can use:
#    rejcomps_man['subject01'] = dict(eeg=[12], meg=[7])

def default_reject_comps():
    return dict(meg=[], eeg=[])

rejcomps_man = defaultdict(default_reject_comps)

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
#
# >>> decoding_conditions = [('Auditory', 'Visual'), ('Left', 'Right')]

decoding_conditions = [('incoherent', 'coherent')]

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

time_frequency_conditions = ['coherent']

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

# base_fname_trans = '{subject}_' + study_name + '_raw-trans.fif'
base_fname_trans = '{subject}-trans.fif'

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

if not os.path.isdir(study_path):
    os.mkdir(study_path)

if not os.path.isdir(subjects_dir):
    os.mkdir(subjects_dir)

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
