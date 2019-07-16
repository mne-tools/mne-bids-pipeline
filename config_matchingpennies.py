from bids import BIDSLayout

study_name = 'eeg_matchingpennies'
task = 'matchingpennies'
kind = 'eeg'
plot = False
bids_root = '/Users/alex/work/data/osfstorage/eeg_matchingpennies'
layout = BIDSLayout(bids_root)
subjects_list = layout.get(return_type='id', target='subject')
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
decoding_conditions = [('left', 'right')]
use_ssp = False
use_ica = False
