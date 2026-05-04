"""osf.io: Matchingpennies EEG.

See [OSF](https://osf.io/download/8rbfk) for more information.
"""

bids_root = "~/mne_data/eeg_matchingpennies"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/eeg_matchingpennies"

ignore_warnings = [
    r"Did not find any eeg\.json associated with sub-",
    "HED annotations were detected but could not be parsed",
    r"Unable to map the following column\(s\)",  # handedness
]

subjects = ["05"]
task = "matchingpennies"
ch_types = ["eeg"]
interactive = False
reject = {"eeg": 150e-6}
conditions = ["raised-left", "raised-right"]
contrasts = [("raised-left", "raised-right")]
decode = True

interpolate_bads_grand_average = False

l_freq = None
h_freq = 100
zapline_fline = 50
zapline_iter = False

# To speed up processing, crop the runs to the first ~10 minutes (rather than all ~31)
crop_runs = (0, 600)
