"""tDCS EEG."""

import numpy as np
import pandas as pd

bids_root = "~/mne_data/ds001810"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds001810"

task = "attentionalblink"
interactive = False
ch_types = ["eeg"]
eeg_template_montage = "biosemi64"
reject = dict(eeg=100e-6)
baseline = (None, 0)
conditions = ["61450", "61511"]
contrasts = [("61450", "61511")]
decode = True
decoding_n_splits = 3  # only for testing, use 5 otherwise

l_freq = 0.3

subjects = ["01"]
sessions = "all"

interpolate_bads_grand_average = False
n_jobs = 4

epochs_custom_metadata = {
    "ses-anodalpost": pd.DataFrame({"ones": np.ones(253)}),
    "ses-anodalpre": pd.DataFrame({"ones": np.ones(268)}),
    "ses-anodaltDCS": pd.DataFrame({"ones": np.ones(269)}),
    "ses-cathodalpost": pd.DataFrame({"ones": np.ones(290)}),
    "ses-cathodalpre": pd.DataFrame({"ones": np.ones(267)}),
    "ses-cathodaltDCS": pd.DataFrame({"ones": np.ones(297)}),
}  # number of rows are hand-set
