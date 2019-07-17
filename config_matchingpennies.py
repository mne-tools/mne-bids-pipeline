"""Configuration file for the eeg_matchingpennies dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_matchingpennies" to overwrite `config.py` with the values specified
below.

Download the eeg_matchingpennies dataset from OSF: https://osf.io/cj2dr/

"""
import os

from bids import BIDSLayout

study_name = 'eeg_matchingpennies'
task = 'matchingpennies'
kind = 'eeg'
plot = False
user = os.environ['USER']
if user == 'gramfort':
    bids_root = '/Users/alex/work/data/osfstorage/eeg_matchingpennies'
elif user == 'stefanappelhoff':
    bids_root = '/home/stefanappelhoff/Downloads/eeg_matchingpennies'
else:
    raise ValueError('bids_root not specified for user "{}"'.format(user))
layout = BIDSLayout(bids_root)
subjects_list = layout.get(return_type='id', target='subject')
reject = {'eeg': 150e-6}
conditions = ['left', 'right']
decoding_conditions = [('left', 'right')]
use_ssp = False
use_ica = False
