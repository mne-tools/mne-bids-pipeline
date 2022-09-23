"""Mobile brain body imaging (MoBI) gait adaptation experiment.

See ds001971 on OpenNeuro: https://github.com/OpenNeuroDatasets/ds001971
"""


study_name = 'ds001971'
bids_root = '~/mne_data/ds001971'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds001971'

task = 'AudioCueWalkingStudy'
interactive = False
ch_types = ['eeg']
reject = {'eeg': 150e-6}
conditions = ['AdvanceTempo', 'DelayTempo']

subjects = ['001']
runs = ['01']
