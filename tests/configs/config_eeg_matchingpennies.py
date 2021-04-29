"""
Matchingpennies EEG experiment
"""

study_name = 'eeg_matchingpennies'
bids_root = '~/mne_data/eeg_matchingpennies'

subjects = ['05']
task = 'matchingpennies'
ch_types = ['eeg']
interactive = False
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
contrasts = [('right', 'left')]
decode = True

interpolate_bads_grand_average = False
