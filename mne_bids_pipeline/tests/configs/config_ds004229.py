"""
Single-subject infant dataset for testing maxwell_filter with movecomp.

https://openneuro.org/datasets/ds004229
"""
# import mne
# import numpy as np

# 1. Get it back to working without movecomp
# 2. Add things back one by one until it's good
study_name = "amnoise"
bids_root = "~/mne_data/ds004229"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds004229"

task = "amnoise"
crop_runs = (0.0, 300.0)

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_cal_fname = bids_root + "/derivatives/meg_derivatives/sss_cal.dat"
mf_ctc_fname = bids_root + "/derivatives/meg_derivatives/ct_sparse.fif"
# mf_destination = mne.transforms.translation(  # rotate 30 deg and then translate 3 cm
#    z=0.03
# ) @ mne.transforms.rotation(x=np.deg2rad(-30))
# mf_mc = False  # TODO: breaks stuff! (maybe a bad channel?)
# mf_int_order = 6  # lower for smaller heads
# mf_mc_t_step_min = 1.0  # just for speed!
# mf_mc_t_window = 0.2
ch_types = ["meg"]

l_freq = None
h_freq = 40.0

# Epochs
epochs_tmin = -0.2
epochs_tmax = 6
reject = {"grad": 4000e-13, "mag": 10000e-15}  # fT/cm, fT
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["auditory"]

# Decoding
decode = False

# Noise estimation
noise_cov = "emptyroom"
