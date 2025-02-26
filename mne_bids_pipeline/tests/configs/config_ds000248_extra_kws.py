"""MNE Sample Data: M/EEG combined processing."""

import mne
import mne_bids

n_jobs = 2

bids_root = "~/mne_data/ds000248"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds000248_extra_kws"
subjects_dir = f"{bids_root}/derivatives/freesurfer/subjects"

subjects = ["01"]
rename_events = {"Smiley": "Emoji", "Button": "Switch"}
conditions = ["Auditory", "Visual", "Auditory/Left", "Auditory/Right"]


ch_types = ["meg", "eeg"]
mf_reference_run = "01"
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True

mf_cal_missing = "warn"
mf_ctc_missing = "warn"


# Extra keyword arguments for find_bad_channels ------------------
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
# ---------------------------------------------------------------