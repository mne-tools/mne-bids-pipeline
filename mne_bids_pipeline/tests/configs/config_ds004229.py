"""Single-subject infant dataset for testing maxwell_filter with movecomp.

https://openneuro.org/datasets/ds004229
"""

import mne
import numpy as np

bids_root = "~/mne_data/ds004229"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds004229"

task = "amnoise"
crop_runs = (300.0, 600.0)  # 5 minutes from the middle of the recording for speed

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_destination = mne.transforms.translation(  # rotate backward and move up
    z=0.055,
) @ mne.transforms.rotation(x=np.deg2rad(-15))
mf_mc = True
mf_destination = "twa"
mf_st_duration = 10
mf_int_order = 6  # lower for smaller heads
mf_mc_t_step_min = 0.5  # just for speed!
mf_mc_t_window = 0.2  # cleaner cHPI filtering on this dataset
mf_filter_chpi = True  # even though we lowpass, set to True for test coverage
mf_mc_rotation_velocity_limit = 30.0  # deg/s for annotations
mf_mc_translation_velocity_limit = 20e-3  # m/s
mf_esss = 8
mf_esss_reject = {"grad": 10000e-13, "mag": 40000e-15}
ch_types = ["meg"]

# test extra kws
find_bad_channels_extra_kws = {
    "ignore_ref": True,
}
mf_extra_kws = {
    "ignore_ref": True,
}
notch_extra_kws = {
    "method": "spectrum_fit",
}
bandpass_extra_kws = {
    "fir_window": "blackman",
}


l_freq = None
h_freq = 40.0

# SSP and peak-to-peak rejection
spatial_filter = "ssp"
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=2, n_grad=2)
ssp_ecg_channel = "MEG0113"  # ECG channel is not hooked up in this dataset
reject = ssp_reject_ecg = {"grad": 2000e-13, "mag": 5000e-15}

# Epochs
epochs_tmin = -0.2
epochs_tmax = 1
epochs_decim = 6  # 1200->200 Hz
baseline = (None, 0)
report_add_epochs_image_kwargs = {
    "grad": {"vmin": 0, "vmax": 1e13 * reject["grad"]},  # fT/cm
    "mag": {"vmin": 0, "vmax": 1e15 * reject["mag"]},  # fT
}

# Conditions / events to consider when epoching
conditions = ["auditory"]

# Decoding
decode = False

# Noise estimation
noise_cov = "emptyroom"
