"""
tDCS EEG
"""

study_name = 'ds001810'
bids_root = '~/mne_data/ds001810'

task = 'attentionalblink'
interactive = False
ch_types = ['eeg']
eeg_template_montage = 'biosemi64'
reject = dict(eeg=100e-6)
baseline = (None, 0)
conditions = ['61450', '61511']
contrasts = [('61450', '61511')]
decode = True

l_freq = 0.3
use_ssp = False

subjects = ['01', '02', '03', '04', '05']
sessions = ['anodalpre']

interpolate_bads_grand_average = False
