"""
MNE Sample Data: ICA
"""
study_name = 'MNE "sample" dataset'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000248_ica'

ch_types = ['meg']
data_type = 'meg'

subjects = ['01']
task = 'audiovisual'
l_freq = 0.3
h_freq = 40.0

conditions = ['Auditory/Left',
              'Auditory/Right',
              'Visual/Left',
              'Visual/Right']

epochs_tmin = -0.2
epochs_tmax = 0.5
baseline = (None, 0)
ica_reject = dict(mag=3000e-15,
                  grad=3000e-13)

spatial_filter = 'ica'
ica_algorithm = 'extended_infomax'
ica_l_freq = 1.0
ica_n_components = 0.8
ica_max_iterations = 500

interactive = False
