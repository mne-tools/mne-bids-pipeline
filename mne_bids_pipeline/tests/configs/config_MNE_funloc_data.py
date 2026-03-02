"""Funloc data."""

from pathlib import Path

data_root = Path("~/mne_data").expanduser().resolve()
bids_root = data_root / "MNE-funloc-data"
deriv_root = data_root / "derivatives" / "mne-bids-pipeline" / "MNE-funloc-data"
subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
task = "funloc"
ch_types = ["meg", "eeg"]
data_type = "meg"

# filter
l_freq = None
h_freq = 50.0
# maxfilter
use_maxwell_filter: bool = True
crop_runs = (40, 190)
mf_st_duration = 60.0
# SSP
spatial_filter = "ssp"
process_raw_clean = False
ssp_ecg_channel = {"sub-01": "MEG0111", "sub-02": "MEG0141"}
n_proj_eog = dict(n_mag=1, n_grad=1, n_eeg=2)
n_proj_ecg = dict(n_mag=1, n_grad=1, n_eeg=0)


eog_channels = {"default": None, "sub-02": ["EOG061"]}

# Epochs
epochs_tmin = -0.2
epochs_tmax = 0.5
epochs_decim = 5  # 1000 -> 200 Hz
baseline = (None, 0)
conditions = [
    "auditory/standard",
    # "auditory/deviant",
    "visual/standard",
    # "visual/deviant",
]
decode = False
decoding_time_generalization = False
cov_rank = dict(tol_kind="relative", tol=1e-4)

# contrasts
# contrasts = [("auditory", "visual")]
