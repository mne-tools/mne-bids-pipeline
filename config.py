"""
===========
Config file
===========

Configuration parameters for the study. This should be in a folder called
``library/`` inside the ``processing/`` directory.
"""

from distutils.version import LooseVersion
import os
import numpy as np


###############################################################################
# DIRECTORIES
# -----------
# Let's set the `study path`` where the data is stored on your system
study_path = '~/mne-data/MNE-sample-data'


# The ``subjects_dir`` and ``meg_dir`` for reading anatomical and MEG files.

subjects_dir = os.path.join(study_path, 'subjects')
meg_dir = os.path.join(study_path, 'MEG')



###############################################################################
# DEFINE SUBJECTS
# ---------------
#
# A list of ``subject names`` 
# These are the ``nips`` in neurospin lingo

subjects = {'subject_01', 'subject_02','subject_03', 'subject_05', 'subject_06',
            'subject_08', 'subject_09','subject_10', 'subject_11','subject_12', 
            'subject_14', 'subject_15', 'subject_16','subject_17', 'subject_18',
            'subject_19', 'subject_23', 'subject_24','subject_25'}

# ``bad subjects``  that should not be included in the analysis
exclude_subjects = {'subject_01', 'subject_09', 'subject_24'}


###############################################################################
# BAD CHANNELS
# ------------
#
# ``bad channels``, to be removed before maxfilter is applied
# you either get them from your recording notes, or from visualizing the data
dict(   subject_01 = 
                        dict(exclude = ['MEG0213','MEG1711','EEG034','EEG035']),
        subject_02 = 
                        dict(exclude = ['MEG0213', 'MEG1233','EEG036']),
        subject_03 = 
                        dict(exclude = ['MEG0213','MEG1512'])
                        )

###############################################################################
# DEFINE ADDITIONAL CHANNELS
# ------------
# 
# Here you name/ replace  extra channels that were recorded, for instance EOG, ECG
# ``set_channel_types`` defines types of channels 
# example : set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog', 'EEG063': 'ecg', 'EEG064': 'misc'}
set_channel_types = None

# ``rename_channels`` rename channels
# example : rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062', 'EEG063': 'ECG063'}
rename_channels = None


###############################################################################
# FREQUENCY FILTERING
# -------------------
#
# ``h_freq``  : the high-frequency cut-off in the lowpass filtering step. 
# Keep it None if no lowpass filtering should be applied.
h_freq = None

# ``l_freq``  : the low-frequency cut-off in the highpass filtering step. 
# Keep it None if no highpass filtering should be applied.
l_freq = 45.


###############################################################################
# MAXFILTER PARAMETERS
# -------------------
#
# Download the ``cross talk file`` and ``calibration file`` (these are machine specific)
# path: 
# and place them in the study folder

ctc = os.path.join(os.path.dirname(__file__), 'ct_sparse.fif')
cal = os.path.join(os.path.dirname(__file__), 'sss_cal.dat')

# ``st_duration `` : if None, no temporal-spatial filtering is applied during MaxFilter,
# otherwise, put a float that speficifies the buffer duration in seconds, 
# Elekta default = 10s, meaning it acts like a 0.1 Hz highpass filter 
st_duration = 30.


###############################################################################
# RESAMPLING
# ----------
#
# ``resample_sfreq``  : a float that specifies at which sampling frequency 
# the data should be resampled. If None then no resampling will be done.
resample_sfreq = 256.


# ``decimate`` : integer that says how much to decimate data at the epochs level. 
# It is typically an alternative to the `resample_sfreq` parameter.
decimate = None


###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
#  ``reject`` : the default rejection limits to make some epochs as bads. 
# This allows to remove strong transient artifacts. 
# **Note**: these numbers tend to vary between subjects.
reject = dict(meg = dict(mag=10e-12, grad=500e-12), 
                eeg = dict(eeg=3000e-6))


###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float that gives the start time before event of an epoch.
tmin = -1.

#  ``tmax`` : float that gives the end time after event of an epochs.
tmax = 2.

# ``baseline`` : tuple that specifies how to baseline the epochs; if None, 
# no baseline is applied

baseline = [-.1, 0.]

#  `event_id`` : python dictionary that maps events (trigger/marker values) 
# to conditions. E.g. `event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}`
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}


###############################################################################
# ICA parameters
# --------------
#- ``runica`` : boolean that says if ICA should be used or not.
runica = True


###############################################################################
# SOURCE SPACE PARAMETERS
# --------
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
# l_trans_bandwidth = 

#  ``h_trans_bandwidth`` : float that specifies the transition bandwidth of the 
# lowpass filter. By default it's `'auto'` and uses default mne parameters.
# h_trans_bandwidth = 


#  ``N_JOBS`` : an integer that specifies how many subjects you want to run in parallel.
# N_JOBS = 1


###############################################################################
# Some mapping betwen filenames for bad sensors and subjects

map_subjects = {1: 'subject_01', 2: 'subject_02', 3: 'subject_03',
                4: 'subject_05', 5: 'subject_06', 6: 'subject_08',
                7: 'subject_09', 8: 'subject_10', 9: 'subject_11',
                10: 'subject_12', 11: 'subject_14', 12: 'subject_15',
                13: 'subject_16', 14: 'subject_17', 15: 'subject_18',
                16: 'subject_19', 17: 'subject_23', 18: 'subject_24',
                19: 'subject_25'}


ylim = {'eeg': [-10, 10], 'mag': [-300, 300], 'grad': [-80, 80]}

annot_kwargs = dict(fontsize=12, fontweight='bold',
                    xycoords="axes fraction", ha='right', va='center')


