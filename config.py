"""
===========
Config file
===========

Configuration parameters for the study. This should be in a folder called
``library/`` inside the ``processing/`` directory.
"""

import os
import numpy as np
from mne.datasets import sample


# let the scripts generate plots or not
# execute %matplotlib qt in your command line once to show the figures in
# separate windows

plot = True
# execute %matplotlib qt 
# in the command line to get the plots in extra windows

###############################################################################
# DIRECTORIES
# -----------
# Let's set the `study path`` where the data is stored on your system
# study_path = '../MNE-sample-data/'
# on windows: study_path = '\Users\sophie\repos\ExampleData\'
study_path = '/Users/sophie/repos/ExampleData/'

# The ``subjects_dir`` and ``meg_dir`` for reading anatomical and MEG files.
subjects_dir = os.path.join(study_path, 'subjects')
meg_dir = os.path.join(study_path, 'MEG')

###############################################################################
# SUBJECTS / RUNS
# ---------------
#
# The MEG-data need to be stored in a folder
# named my_study_path/MEG/my_subject/

# This is the name of your experimnet
study_name = 'Localizer'

# To define the subjects, we use a list with all the subject names. Even if its
# a single subject, it needs to be set up as a list with a single element,
# as in the example

subjects_list = ['SB01'] # ,'SB02', 'SB03'
# subjects_list = ['subject_01', 'subject_02', 'subject_03', 'subject_05',
#                  'subject_06', 'subject_08', 'subject_09', 'subject_10',
#                  'subject_11', 'subject_12', 'subject_14']

# ``bad subjects`` that should not be excluded from the above
exclude_subjects = []  # ['subject_01']


# Define the names of your ``runs``
# The naming should be consistant over subjects.
# put the number of runs you ideally expect to have per subject
# the scripts will issue a warning if there are less
# leave empty if there is just one file
runs = [''] # ['run01', 'run02']

# does the data have EEG?

eeg = False # True

# This generates the name for all files
# with the names specified above
# normally you should not have to touch this
base_fname = '{subject}_' + study_name + '{extension}.fif'

###############################################################################
# BAD CHANNELS
# ------------
#
# ``bad channels``, to be removed before maxfilter is applied
# you either get them from your recording notes, or from visualizing the data
# Use the simple dict if you don't have runs, and the dict(dict) if you have runs

bads = dict(SB01=['MEG1723','MEG1722'],
            SB02=[],
            SB03=[],
            )

#  if you have multiple runs, you need to define bad channels per run
# bads = dict(SB01=dict(run01=['MEG 2443', 'EEG 053'],
#                         run02=['MEG 2443', 'EEG 053', 'EEG 013']))

###############################################################################
# DEFINE ADDITIONAL CHANNELS
# --------------------------
#
# Here you name/ replace  extra channels that were recorded, for instance EOG, ECG
# ``set_channel_types`` defines types of channels
# example :
# set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog', 'EEG063': 'ecg', 'EEG064': 'misc'}
set_channel_types = None

# ``rename_channels`` rename channels
#
# example :
# rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062', 'EEG063': 'ECG063'}
rename_channels = None

###############################################################################
# FREQUENCY FILTERING
# -------------------
#

# ``l_freq``  : the low-frequency cut-off in the highpass filtering step.
# Keep it None if no highpass filtering should be applied.
l_freq = 1.

# ``h_freq``  : the high-frequency cut-off in the lowpass filtering step.
# Keep it None if no lowpass filtering should be applied.
h_freq = 40.

###############################################################################
# MAXFILTER PARAMETERS
# -------------------
#
# Download the ``cross talk file`` and ``calibration file`` (these are machine specific)
# path:
# and place them in the study folder
mf_ctc_fname = os.path.join(study_path, 'system_calibration_files', 'NeuroSpin_ct_sparse.fif')
mf_cal_fname = os.path.join(study_path, 'system_calibration_files',  'NeuroSpin_sss_cal.dat')

# ``mf_reference_run `` : defines the reference run used to adjust the head position for
# all other runs
mf_reference_run = 0  # take 1st run as reference for head position

# Set the origin for the head position
mf_head_origin = 'auto'

# ``mf_st_duration `` : if None, no temporal-spatial filtering is applied during MaxFilter,
# otherwise, put a float that speficifies the buffer duration in seconds,
# Elekta default = 10s, meaning it acts like a 0.1 Hz highpass filter
mf_st_duration = None

###############################################################################
# RESAMPLING
# ----------
#
# ``resample_sfreq``  : a float that specifies at which sampling frequency
# the data should be resampled. If None then no resampling will be done.
resample_sfreq =  500. # None


# ``decim`` : integer that says how much to decimate data at the epochs level.
# It is typically an alternative to the `resample_sfreq` parameter that
# can be used for resampling raw data. 1 means no decimation.
decim = 1

###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
#  ``reject`` : the default rejection limits to make some epochs as bads.
# This allows to remove strong transient artifacts.
# If you want to reject and retrieve blinks later, e.g. with ICA, don't specify
# a value for the eog channel (see examples below).
# Make sure to include values for eeg if you have eeg data

# **Note**: these numbers tend to vary between subjects.
# Examples:
# reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
# reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 200e-6}
# reject = None

reject = {'grad': 4000e-13, 'mag': 4e-12}

###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float that gives the start time before event of an epoch.
tmin = -0.6

#  ``tmax`` : float that gives the end time after event of an epochs.
tmax = 1.5

trigger_offset = -0.0416
# XXX forward/delay all triggers by this value

# ``baseline`` : tuple that specifies how to baseline the epochs; if None,
# no baseline is applied

baseline = (-.6, -.1) # (None, 0.)

# stimulus channel, which contains the events
stim_channel = 'STI101'  # 'STI014'# None

# minimal duration of the events you want to extract
min_event_duration = 0.002

#  `event_id`` : python dictionary that maps events (trigger/marker values)
# to conditions. E.g. `event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}`
# event_id = {'Onset': 4}
# conditions = ['Onset']

event_id = {'incoherent_1': 33, 'incoherent_2': 35,
            'coherent_down': 37, 'coherent_up': 39}
conditions = ['incoherent_1', 'incoherent_2', 'coherent_down', 'coherent_up']

###############################################################################
# ICA PARAMETERS
# --------------
# ``runica`` : boolean that says if ICA should be used or not.
runica = True

rejcomps_man = dict(SB01=dict(meg=[],
                                eeg=[]))


###############################################################################
# DECODING
# --------------
#
# decoding_conditions should be a list of conditions to be classified.
# For example 'Auditory' vs. 'Visual' as well as
# 'Auditory/Left' vs 'Auditory/Right'
decoding_conditions = [('Auditory/Left', 'Auditory/Right'),
                       ('Auditory', 'Visual')]
decoding_metric = 'roc_auc'
decoding_n_splits = 5

###############################################################################
# TIME-FREQUENCY
# --------------
#
time_frequency_conditions = ['Auditory/Left']

###############################################################################
# SOURCE SPACE PARAMETERS
# -----------------------
#

spacing = 'oct6'
mindist = 5
smooth = 10

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

if not os.path.isdir(study_path):
    os.mkdir(study_path)

if not os.path.isdir(subjects_dir):
    os.mkdir(subjects_dir)

###############################################################################
# ADVANCED
# --------
#
# ``l_trans_bandwidth`` : float that specifies the transition bandwidth of the
# highpass filter. By default it's `'auto'` and uses default mne parameters.
l_trans_bandwidth = 'auto'

#  ``h_trans_bandwidth`` : float that specifies the transition bandwidth of the
# lowpass filter. By default it's `'auto'` and uses default mne parameters.
h_trans_bandwidth = 'auto'

#  ``N_JOBS`` : an integer that specifies how many subjects you want to run in parallel.
N_JOBS = 1

random_state = 42
