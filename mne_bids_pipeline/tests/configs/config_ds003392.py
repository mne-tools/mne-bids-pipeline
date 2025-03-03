"""hMT+ Localizer."""

import os

from mne_bids import BIDSPath

bids_root = "~/mne_data/ds003392"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds003392"

subjects = ["01"]

task = "localizer"
# usually a good idea to use True, but we know no bads are detected for this dataset
find_flat_channels_meg = False
find_noisy_channels_meg = False
use_maxwell_filter = True
mf_extra_kws = {"bad_condition": "warning"}
ch_types = ["meg"]

mf_cal_missing = "warn"
mf_ctc_missing = "warn"


# Remove meg_calibration_fpath
calibration_path = BIDSPath(
    subject=subjects[0],
    suffix="meg",
    datatype="meg",
    root=bids_root,
).meg_calibration_fpath
try:
    os.remove(calibration_path)
    print(f"File '{calibration_path}' successfully removed.")
except FileNotFoundError:
    print(f"File '{calibration_path}' not found.")

# Remove meg_crosstalk_fpath
crosstalk_path = BIDSPath(
    subject=subjects[0],
    suffix="meg",
    datatype="meg",
    root=bids_root,
).meg_crosstalk_fpath
try:
    os.remove(crosstalk_path)
    print(f"File '{crosstalk_path}' successfully removed.")
except FileNotFoundError:
    print(f"File '{crosstalk_path}' not found.")


l_freq = 1.0
h_freq = 40.0
raw_resample_sfreq = 250
crop_runs = (0, 180)

# Artifact correction.
spatial_filter = "ica"
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

# Decoding
decode = True
decoding_time_generalization = True
decoding_time_generalization_decim = 4
contrasts = [("incoherent", "coherent")]
decoding_csp = True
decoding_csp_times = []
decoding_csp_freqs = {
    "alpha": (8, 12),
}

# Noise estimation
noise_cov = "emptyroom"
