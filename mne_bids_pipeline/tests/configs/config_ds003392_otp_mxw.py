"""ds003392: hMT+ Localizer.

See [OpenNeuro](https://openneuro.org/datasets/ds003392) for more information.
This config tests implementation of oversampled temporal projection followed by maxwell
filtering and subsequent preprocessing steps.
"""

# from mne.transforms import translation

bids_root = "~/mne_data/ds003392_otp_mxw"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds003392_otp_mxw"
ignore_warnings = [
    "Internal Active Shielding data",  # until MNE-BIDS releases a fix for ERM finding
]
subjects = ["01"]

task = "localizer"

# use oversampled temporal projection to clean sensor noise
use_otp_denoising = True
otp_duration = 10.0
# for OTP testing purposes, set to True
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_extra_kws = {"bad_condition": "warning"}
ch_types = ["meg"]
mf_esss = 1
# translation args should be x, y, z as int
# mf_destination = translation(z=0.04)

mf_cal_missing = "warn"
mf_ctc_missing = "warn"

l_freq = 1.0
h_freq = 40.0
raw_resample_sfreq = 250
crop_runs = (0, 20)

# Artifact correction
spatial_filter = "ica"
process_raw_clean = False
ica_algorithm = "picard-extended_infomax"
ica_max_iterations = 1000
ica_l_freq = 1.0
ica_n_components = 0.99

# Epochs
epochs_tmin = -0.2
epochs_tmax = 1.0
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["coherent", "incoherent"]
