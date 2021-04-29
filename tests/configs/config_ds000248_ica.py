"""
MNE Sample Data: ICA
"""
study_name = 'MNE "sample" dataset'
bids_root = '~/mne_data/ds000248_ica'
ch_types = ['meg']
data_type = 'meg'

subjects = ['01']
task = 'audiovisual'
run = '01'
l_freq = 0.3
h_freq = 40.0

conditions = ['Auditory/Left',
              'Auditory/Right',
              'Visual/Left',
              'Visual/Right']

tmin = -0.2
tmax = 0.5
baseline = (None, 0)
reject = dict(mag=3000e-15,
              grad=3000e-13)


spatial_filter = 'ica'
ica_algorithm = 'extended_infomax'
ica_l_freq = 1.0
ica_n_components = 0.8
ica_max_iterations = 500

interactive = False
