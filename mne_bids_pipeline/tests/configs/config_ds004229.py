"""
Single-subject infant dataset for testing maxwell_filter with movecomp.

https://openneuro.org/datasets/ds004229
"""
study_name = 'amnoise'
bids_root = '~/mne_data/ds004229'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds004229'

task = 'amnoise'
crop_runs = (0., 300.)
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_cal_fname = bids_root + '/derivatives/meg_derivatives/sss_cal.dat'
mf_ctc_fname = bids_root + '/derivatives/meg_derivatives/ct_sparse.fif'
ch_types = ['meg']

l_freq = None
h_freq = 40.

# Epochs
epochs_tmin = -0.2
epochs_tmax = 6
epochs_decim = 5
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["auditory"]

# Decoding
decode = False

# Noise estimation
noise_cov = 'emptyroom'
