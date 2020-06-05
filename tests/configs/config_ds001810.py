"""Configuration file for the ds001810 dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_ds001810" to overwrite `config.py` with the values specified
below.

Download ds001810 from OpenNeuro: https://github.com/OpenNeuroDatasets/ds001810

export MNE_BIDS_STUDY_CONFIG=config_ds001810
export BIDS_ROOT=~/mne_data/ds001810

"""


study_name = 'ds001810'
task = 'attentionalblink'
interactive = False
ch_types = ['eeg']
reject = {'eeg': 150e-6}
conditions = ['61510', '61511']
contrasts = [('61510', '61511')]
decode = True
use_ssp = False
use_ica = False

subjects_list = ['01']
sessions = ['anodalpre']

interpolate_bads_grand_average = False
