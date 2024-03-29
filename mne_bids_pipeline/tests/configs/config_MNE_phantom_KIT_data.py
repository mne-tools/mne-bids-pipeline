"""
KIT phantom data.

https://mne.tools/dev/documentation/datasets.html#kit-phantom-dataset
"""

bids_root = "~/mne_data/MNE-phantom-KIT-data"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/MNE-phantom-KIT-data"
task = "phantom"
ch_types = ["meg"]

# Preprocessing
l_freq = None
h_freq = 40.0
regress_artifact = dict(
    picks="meg", picks_artifact=["MISC 001", "MISC 002", "MISC 003"]
)

# Epochs
epochs_tmin = -0.08
epochs_tmax = 0.18
epochs_decim = 10  # 2000->200 Hz
baseline = (None, 0)
conditions = ["dip01", "dip13", "dip25", "dip37", "dip49"]

# Decoding
decode = True  # should be very good performance
