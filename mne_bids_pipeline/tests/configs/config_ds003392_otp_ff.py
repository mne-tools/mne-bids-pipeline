"""ds003392: hMT+ Localizer (with OTP).

See [OpenNeuro](https://openneuro.org/datasets/ds003392) for more information.
This config tests implementation of oversampled temporal projection, skipping maxwell
filtering, followed by frequency filtering and subsequent preprocessing steps.
"""

bids_root = "~/mne_data/ds003392"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds003392_otp_ff"
ignore_warnings = [
    "Internal Active Shielding data",  # until MNE-BIDS releases a fix for ERM finding
]
subjects = ["01"]

task = "localizer"

# use oversampled temporal projection to clean sensor noise
use_otp_denoising = True
otp_duration = 10.0
# for OTP testing purposes, set bad channel detection to True
find_flat_channels_meg = True
find_noisy_channels_meg = True
# Don't use maxwell filtering so we can test appropriate deriv passing from OTP to
# frequency filter
use_maxwell_filter = False
ch_types = ["meg"]
# Still pass Maxwell option for denoising step
mf_extra_kws = {"bad_condition": "warning"}
mf_cal_missing = "warn"
mf_ctc_missing = "warn"

l_freq = 1.0
h_freq = 40.0
raw_resample_sfreq = 250
crop_runs = (0, 20)

# Epochs
epochs_tmin = -0.2
epochs_tmax = 1.0
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["coherent", "incoherent"]
