"""
MNE "sample" dataset: here we only process EEG data and check ICA.
"""
study_name = 'MNE "sample" dataset'
ch_types = ['eeg']
data_type = 'meg'

subjects_list = ['01']
task = 'audiovisual'
run = '01'
l_freq = 0.3

conditions = ['Auditory/Left',
              'Auditory/Right',
              'Visual/Left',
              'Visual/Right']

tmin = -0.3
tmax = 0.7
baseline = (None, 0)

use_ssp = False
use_ica = True
ica_n_components = 15
ica_l_freq = 1.
ica_reject_components = 'auto'

interactive = False
