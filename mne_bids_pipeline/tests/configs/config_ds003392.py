"""
hMT+ Localizer
"""
study_name = 'localizer'
bids_root = '~/mne_data/ds003392'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds003392'

subjects = ['01']

task = 'localizer'
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
ch_types = ['meg']

l_freq = 1.
h_freq = 40.
raw_resample_sfreq = 250
crop_runs = (0, 180)

# Artifact correction.
spatial_filter = 'ica'
ica_max_iterations = 500
ica_l_freq = 1.
ica_n_components = 0.99
ica_reject_components = 'auto'

# Epochs
epochs_tmin = -0.2
epochs_tmax = 1.0
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ['coherent', 'incoherent']

# Decoding
decode = True
decoding_time_generalization = True
decoding_time_generalization_decim = 4
contrasts = [('incoherent', 'coherent')]

# Noise estimation
noise_cov = 'emptyroom'
