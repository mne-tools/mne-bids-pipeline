"""Configuration file for the ds001971 dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_ds001971" to overwrite `config.py` with the values specified
below.

Download ds001971 from OpenNeuro: https://github.com/OpenNeuroDatasets/ds001971

export MNE_BIDS_STUDY_CONFIG=config_ds001971
export BIDS_ROOT=~/mne_data/ds001971

"""


study_name = 'ds001971'
task = 'AudioCueWalkingStudy'
interactive = False
ch_types = ['eeg']
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
contrasts = [('left', 'right')]
decode = True
use_ssp = False
use_ica = False


subjects_list = ['001']

runs = ['01']
