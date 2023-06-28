"""
Single-subject infant dataset for testing maxwell_filter with movecomp.

https://openneuro.org/datasets/ds004229
"""
import mne
import numpy as np

# 1. Get it back to working without movecomp -> reject was a problem!
# 2. Add things back one by one until it's good
# 3. Include tSSS?
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
mf_destination = mne.transforms.translation(  # rotate 30 deg and then translate
    y=-0.014,
    z=0.055,
) @ mne.transforms.rotation(x=np.deg2rad(-30))
mf_mc = True
mf_st_duration = 10
mf_int_order = 6  # lower for smaller heads
mf_mc_t_step_min = 1.0  # just for speed!
mf_mc_t_window = 0.2  # cleaner cHPI filtering on this dataset
ch_types = ["meg"]

l_freq = None
h_freq = 40.0

# SSP and peak-to-peak rejection
spatial_filter = "ssp"
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=1, n_grad=1)  # the default but be explicit
ssp_ecg_channel = "MEG0113"  # ECG channel is not hooked up in this dataset
reject = ssp_reject_ecg = {"grad": 2000e-13, "mag": 5000e-15}

# Epochs
epochs_tmin = -0.2
epochs_tmax = 6
epochs_decim = 6  # 1200->200
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["auditory"]

# Decoding
decode = False

# Noise estimation
noise_cov = "emptyroom"
