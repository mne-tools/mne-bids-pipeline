"""Configuration file for the eeg_matchingpennies dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_eeg_matchingpennies" to overwrite `config.py` with the values specified
below.

Download the eeg_matchingpennies dataset from OSF: https://osf.io/cj2dr/

"""

study_name = 'eeg_matchingpennies'
subjects_list = ['05']
task = 'matchingpennies'
ch_types = ['eeg']
interactive = False
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
contrasts = [('right', 'left')]
decode = True
use_ssp = False
use_ica = False

interpolate_bads_grand_average = False
