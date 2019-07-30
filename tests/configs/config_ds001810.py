"""Configuration file for the eeg_matchingpennies dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_ds001810" to overwrite `config.py` with the values specified
below.

Download ds001810 from OpenNeuro: https://github.com/OpenNeuroDatasets/ds001810

"""


study_name = 'ds001810'
task = 'attentionalblink'
kind = 'eeg'
plot = False
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
decoding_conditions = [('left', 'right')]
use_ssp = False
use_ica = False


subjects_list = ['01']
