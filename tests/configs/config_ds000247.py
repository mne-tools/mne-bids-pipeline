"""
OMEGA Resting State Sample Data
"""

study_name = 'ds000247'
bids_root = '~/mne_data/ds000247'

subjects = ['0002']
sessions = ['01']
task = 'rest'

crop_runs = (0, 100)  # to speed up computations

ch_types = ['meg']
spatial_filter = 'ssp'

l_freq = 1.0
h_freq = 40.0

rest_epochs_duration = 10
rest_epochs_overlap = 0

epochs_tmin = 0
baseline = None
