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
contrasts = [("61450", "61511"), ("letter=='a'", "letter=='b'")]
decode = True
decoding_n_splits = 3  # only for testing, use 5 otherwise
decoding_time_decim = 3  # for speed

l_freq = 0.3

subjects = ["01"]
sessions = "all"

interpolate_bads_grand_average = False
n_jobs = 4

epochs_custom_metadata = {
    "ses-anodalpost": pd.DataFrame(
        {
            "ones": np.ones(253),
            "letter": ["a" for x in range(150)] + ["b" for x in range(103)],
        }
    ),
    "ses-anodalpre": pd.DataFrame(
        {
            "ones": np.ones(268),
            "letter": ["a" for x in range(150)] + ["b" for x in range(118)],
        }
    ),
    "ses-anodaltDCS": pd.DataFrame(
        {
            "ones": np.ones(269),
            "letter": ["a" for x in range(150)] + ["b" for x in range(119)],
        }
    ),
    "ses-cathodalpost": pd.DataFrame(
        {
            "ones": np.ones(290),
            "letter": ["a" for x in range(150)] + ["b" for x in range(140)],
        }
    ),
    "ses-cathodalpre": pd.DataFrame(
        {
            "ones": np.ones(267),
            "letter": ["a" for x in range(150)] + ["b" for x in range(117)],
        }
    ),
    "ses-cathodaltDCS": pd.DataFrame(
        {
            "ones": np.ones(297),
            "letter": ["a" for x in range(150)] + ["b" for x in range(147)],
        }
    ),
}  # number of rows are hand-set
